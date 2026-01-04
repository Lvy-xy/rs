import sys
import os
import time
import warnings
import threading
import cv2
import numpy as np
import snap7
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QSizePolicy, QGroupBox, QFrame
)
from ultralytics import YOLO

# ================= PLC 通信工具函数 =================
PLC_POLL_INTERVAL_MS = 50
PLC_RETRY_DELAY_SEC = 0.05
PLC_FAST_TRIGGER_WINDOW_SEC = 0.1

def plc_connect(ip, conn_type, rack=0, slot=1):
    client = snap7.client.Client()
    client.set_connection_type(conn_type)
    try:
        client.connect(ip, rack, slot)
        if client.get_connected():
            print(f"[PLC连接] 成功连接到 PLC：{ip}")
            return client
        else:
            print(f"[PLC连接] 无法连接到 PLC：{ip}")
            return None
    except Exception as e:
        print(f"[PLC连接异常] {e}")
        return None

def plc_con_close(client):
    if client and client.get_connected():
        client.disconnect()
        print("[PLC断开] 成功断开与 PLC 的连接")

def read_word(client, offset, log=True):
    try:
        data = client.db_read(4, offset, 2)
        value = int.from_bytes(data, byteorder='big', signed=True)
        if log:
            print(f"[PLC读取] 读取 DB4.DBW{offset} 的值为 {value}")
        return value
    except Exception as e:
        print(f"[PLC读取异常] 无法读取 DB4.DBW{offset}，错误信息：{e}")
        return 0

def write_word(client, offset, value, max_retries=3):
    value = max(-32768, min(32767, int(value)))
    for attempt in range(max_retries):
        try:
            client.db_write(4, offset, value.to_bytes(2, byteorder='big', signed=True))
            print(f"[PLC写回] 成功写入值 {value} 到 DB4.DBW{offset}")
            return True
        except Exception as e:
            print(f"[PLC写入异常] 无法写入 DB4.DBW{offset}，错误信息：{e}")
            time.sleep(PLC_RETRY_DELAY_SEC)  # 降低重试等待，减少发送延迟
    print(f"[PLC写入失败] 多次尝试写入 DB4.DBW{offset} 仍失败，放弃")
    return False

def write_result(client, result_value, status_value=2, max_retries=3):
    status_value = max(-32768, min(32767, int(status_value)))
    result_value = max(-32768, min(32767, int(result_value)))
    payload = status_value.to_bytes(2, byteorder='big', signed=True) + result_value.to_bytes(
        2, byteorder='big', signed=True
    )
    for attempt in range(max_retries):
        try:
            # DBW0(状态) 与 DBW2(结果) 合并写入，减少一次通讯开销
            client.db_write(4, 0, payload)
            print(f"[PLC写回] 成功写入状态 {status_value} 与结果 {result_value} 到 DB4.DBW0/DBW2")
            return True
        except Exception as e:
            print(f"[PLC写入异常] 无法写入 DB4.DBW0/DBW2，错误信息：{e}")
            time.sleep(PLC_RETRY_DELAY_SEC)
    print("[PLC写入失败] 多次尝试写入 DB4.DBW0/DBW2 仍失败，放弃")
    return False

# ================= 主窗口类（核心功能） =================
class GinsengClassifierGUI(QWidget):
    def __init__(self, model_path, plc_ip):
        super().__init__()
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        # 窗口基础配置
        self.setWindowTitle('🧪 人参识别与筛选实时监控')
        self.resize(1000, 600)
        self.setFixedSize(1000, 600)
        self.move(100, 100)
        self.setStyleSheet('background:#1e1e2f; color:#eee;')

        # 核心组件初始化
        self.model = YOLO(model_path, task='detect')  # 加载YOLO模型（支持.pt/.xml格式）
        self.plc_ip = plc_ip
        self.plc = plc_connect(plc_ip, 2)
        self.plc_connected = self.plc is not None
        self.plc_count = 0
        self.plc_status = ''

        # 类别映射（已修改为“分叉”）
        self.class_names = {
            0: '病斑', 1: '成品人参', 2: '带泥', 3: '分叉',
            4: '磕巴', 5: '烂头', 6: '锈病', 7: '芽孢'
        }
        self.level_counts = {i: 0 for i in range(8)}  # 8类统计初始化

        # 摄像头初始化
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError('无法打开摄像头，请检查设备连接')

        # 并发控制锁（避免重复识别）
        self.infer_lock = threading.Lock()
        self.last_plc_status = None
        self.last_plc_ts = 0.0

        # 初始化UI
        self.init_ui()

        # 双定时器配置（画面刷新+PLC信号检测分离，避免卡顿）
        self.plc_timer = QTimer(self)
        self.plc_timer.timeout.connect(self.check_plc_signal)
        self.plc_timer.start(PLC_POLL_INTERVAL_MS)  # 更快检测PLC信号，降低触发延迟

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_camera_frame)
        self.video_timer.start(30)  # 30ms刷新一次画面（约33fps）

    def init_ui(self):
        # 水平分割器（左侧视频区+右侧信息面板）
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(2)

        # 左侧：摄像头画面显示区
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet('border:3px solid #444; background:#000;')
        splitter.addWidget(self.video_label)

        # 右侧：信息统计面板
        info_frame = QFrame()
        info_frame.setFixedWidth(380)
        info_frame.setStyleSheet('background:#2e2e3e; border-radius:8px;')
        vbox = QVBoxLayout(info_frame)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(15)

        # 面板标题
        title = QLabel('📊 实时统计信息板')
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        vbox.addWidget(title)

        # 各类别统计GroupBox
        self.level_boxes = {}
        for cls_id in sorted(self.level_counts):
            name = self.class_names[cls_id]
            box = QGroupBox(f'{name}: 0')
            box.setFixedHeight(60)
            box.setStyleSheet('QGroupBox{background:#3e3e4e; border:1px solid #555; border-radius:5px;}')
            vbox.addWidget(box)
            self.level_boxes[cls_id] = box

        # 当前识别结果显示
        self.current_box = QGroupBox('当前类别: None')
        self.current_box.setFixedHeight(60)
        self.current_box.setStyleSheet('QGroupBox{background:#3e3e4e; border:1px solid #555; border-radius:5px;}')
        vbox.addWidget(self.current_box)

        # PLC状态显示
        self.plc_box = QGroupBox()
        self.plc_box.setFixedHeight(60)
        self.plc_box.setStyleSheet('QGroupBox{background:#3e3e4e; border:1px solid #555; border-radius:5px;}')
        vbox.addWidget(self.plc_box)
        self.update_plc_status()

        # 操作按钮布局
        btn_layout = QHBoxLayout()
        for txt, clr, fn in [
            ('🔄 重连 PLC', '#28a745', self.reconnect_plc),
            ('全屏', '#0078d7', self.toggle_fullscreen),
            ('退出全屏', '#f0ad4e', lambda: self.setWindowState(self.windowState() & ~Qt.WindowFullScreen)),
            ('退出', '#d70022', self.close),
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            btn.setFixedSize(80, 40)
            btn.setStyleSheet(f'''
                QPushButton {{ background:{clr}; color:#fff; border:none; border-radius:6px; font-size:14px; }}
                QPushButton:hover {{ background:#444; }}
            ''')
            btn_layout.addWidget(btn)
        btn_layout.setSpacing(10)
        vbox.addLayout(btn_layout)
        vbox.addStretch()

        # 分割器比例设置（左侧3:右侧1）
        splitter.addWidget(info_frame)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)

        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    # ================= UI交互功能 =================
    def toggle_fullscreen(self):
        """切换全屏/退出全屏"""
        self.setWindowState(self.windowState() ^ Qt.WindowFullScreen)

    def update_plc_status(self):
        """更新PLC连接状态显示"""
        if self.plc and self.plc.get_connected():
            self.plc_connected = True
            self.plc_status = f'✅ PLC 已连接 | 执行次数: {self.plc_count}'
        else:
            self.plc_connected = False
            self.plc_status = '❌ PLC 未连接'
        self.plc_box.setTitle(self.plc_status)

    def reconnect_plc(self):
        """手动重连PLC"""
        if self.plc:
            plc_con_close(self.plc)
        self.plc = plc_connect(self.plc_ip, 2)
        self.update_plc_status()

    # ================= 核心业务逻辑 =================
    def check_plc_signal(self):
        """检测PLC触发信号（DB4.DBW0=1时启动识别）"""
        if not self.plc_connected:
            print("[PLC信号] 未连接到 PLC，跳过检测")
            return
        status = read_word(self.plc, 0, log=False)
        now = time.time()
        if status != self.last_plc_status:
            print(f"[PLC信号] 当前 DB4.DBW0 状态值为 {status}")
        self.last_plc_status = status
        self.last_plc_ts = now
        # 状态为1且无正在执行的识别时，启动子线程推理
        if status == 1 and not self.infer_lock.locked():
            print("[PLC触发] 状态为1，启动识别子线程")
            threading.Thread(target=self.perform_inference, daemon=True).start()

    def update_camera_frame(self):
        """实时刷新摄像头画面（不阻塞UI）"""
        ret, frame = self.capture.read()
        if not ret:
            return
        # 转换为Qt支持的图像格式并显示
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def perform_inference(self):
        """模型推理核心逻辑（子线程中执行）"""
        with self.infer_lock:  # 加锁防止并发冲突
            if not self.plc_connected:
                print("[识别中止] PLC 已断开连接")
                return

            # 二次校验PLC信号（避免信号中断导致无效推理）
            if self.last_plc_status == 1 and (time.time() - self.last_plc_ts) <= PLC_FAST_TRIGGER_WINDOW_SEC:
                plc_status = 1
            else:
                plc_status = read_word(self.plc, 0)
            if plc_status != 1:
                print("[识别中止] PLC 信号已变更，不再执行识别")
                return

            start_time = time.time()
            # 读取摄像头帧用于推理
            ret, frame = self.capture.read()
            if not ret:
                print("[识别失败] 摄像头读取画面失败")
                write_word(self.plc, 0, 2)  # 写入识别失败状态
                return

            # 模型推理（指定输入尺寸640，与训练一致）
            img = frame
            res = self.model([img], imgsz=640)[0]

            # 解析推理结果（筛选置信度最高的目标）
            best_cls, best_conf = None, 0
            for box, conf, cls in zip(
                    res.boxes.xyxy, res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)
            ):
                # 绘制目标框
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制类别+置信度标签
                if conf > 0.1:
                    label = self.class_names.get(cls, str(cls))
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"[目标检测] 类别：{label}，置信度：{conf:.2f}，位置：({x1},{y1})-({x2},{y2})")

                # 更新置信度最高的目标
                if conf > 0.1 and conf > best_conf:
                    best_cls, best_conf = int(cls), conf

            # 处理识别结果（含容错逻辑）
            if best_cls is not None and best_cls in self.class_names:
                # 有有效目标：更新统计并写入PLC
                self.level_counts[best_cls] += 1
                name = self.class_names[best_cls]
                self.level_boxes[best_cls].setTitle(f'{name}: {self.level_counts[best_cls]}')
                self.current_box.setTitle(f'当前类别: {name}')
                print(f"[识别结果] 最优类别：{name}（ID:{best_cls}），写入PLC值 {best_cls+1}")
                write_result(self.plc, best_cls + 1)
            else:
                # 无有效目标：默认归类为成品人参（工业场景适配）
                self.level_counts[1] += 1
                default_name = self.class_names[1]
                self.level_boxes[1].setTitle(f'{default_name}: {self.level_counts[1]}')
                self.current_box.setTitle(f'当前类别: {default_name}（默认）')
                print(f"[识别结果] 无有效目标，默认归类为{default_name}，写入PLC值 2")
                write_result(self.plc, 2)

            # 更新PLC执行次数和状态
            self.plc_count += 1
            self.update_plc_status()

            print("[PLC反馈] 识别完成，已写入状态值 2 与识别结果")

            # 刷新带标注的画面
            qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_BGR888)
            pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)

            # 打印识别耗时
            end_time = time.time()
            print(f"[识别耗时] 本次识别总耗时：{int((end_time - start_time) * 1000)} 毫秒\n")

    # ================= 资源释放 =================
    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        self.plc_timer.stop()
        self.video_timer.stop()
        self.capture.release()
        plc_con_close(self.plc)
        print("[程序退出] 所有资源已释放")
        event.accept()

# ================= 程序入口 =================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 核心配置（已修改为你的实际路径和PLC IP）
    GUI = GinsengClassifierGUI(
        model_path=r'C:\Users\user\PycharmProjects\PythonProject\rsUI\rs\model\best(1).pt',  # 模型路径（支持.pt或OpenVINO的.xml）
        plc_ip='192.168.1.10'  # PLC实际IP，需根据设备修改
    )
    GUI.show()
    sys.exit(app.exec_())
