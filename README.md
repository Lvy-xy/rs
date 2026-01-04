# yolo_rs Web + PLC 集成说明

本项目提供浏览器端摄像头截帧 + YOLO 目标检测，并集成 PLC 通信（DB4：DBW0 触发、DBW2 写结果）。启动页 `/`，识别页 `/app`。

## 快速开始
- 安装依赖：`python -m pip install -r requirements.txt`
- 运行服务：`python app.py`，浏览器访问 `http://localhost:5000/`
- 允许摄像头权限，点击“启动设备”进入识别页。

## 环境变量
- `PLC_IP`（默认 `192.168.0.17`）
- `PLC_DB`（默认 `4`）
- `PLC_RACK`（默认 `0`）
- `PLC_SLOT`（默认 `1`）
- `PLC_CONN_TYPE`（默认 `2`）
- `YOLO_CONF`（默认 `0.1`，最低置信度阈值）
未配置或无 PLC 时，仍可截帧检测，但不会写入 PLC。

## 模型与类别
- 默认模型：`model/yolo_rs.pt` 自动读取 names；失败时回退中文 8 类：病斑、成品、带泥、分叉、磕疤、烂头、锈、芽孢。
- 可将其他 `.pt` 放入 `model/`，前端下拉切换。

## 前端交互
- 左侧实时预览，点击“截取当前帧并识别”后右侧显示带框结果。
- PLC 状态徽标：未启动/已连接(IP)/未连接，自动轮询。
- 分类累计、重置、JSON 导出、暂停预览。

## PLC 逻辑（对齐原 main_pro3dan）
- 进入 `/app` 自动调用 `POST /plc/start` 尝试连接。
- 触发：PLC 连上时，仅当 DB4.DBW0 == 1 才接受识别；否则返回 409；前端轮询 DBW0 并在 ==1 时自动截帧识别（按钮仍可手动触发）。
- 写回：识别完成写 DB4.DBW2 = 类别 ID（1-based），DB4.DBW0 = 2，并累计 exec_count。

## API 摘要
- `POST /plc/start`：建立 PLC 连接并返回状态。
- `GET /plc/status`：查询 PLC 连接/错误/执行次数。
- `POST /detect`：{ image: base64, model } → 检测结果 + PLC 状态（连接 PLC 时需 DBW0 == 1）。

## 目录
- `config.py`：全局配置（路径、模型、PLC 参数），支持环境变量覆盖。
- `app.py`：Flask 服务、YOLO 推理、PLC 通信。
- `templates/home.html`：启动页。
- `templates/index.html`：识别页（摄像头 + 统计 + PLC 状态）。
- `model/`：YOLO 权重。
- `images/index.jpg`：启动页封面。
