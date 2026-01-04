import base64
import io
import time
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request
from PIL import Image

from . import config
from .model import ModelManager, Detection, CLASS_META
from .plc import PLCManager

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATE_DIR),
    static_folder=str(config.STATIC_DIR),
    static_url_path="/images",
)

model_manager = ModelManager(config.MODEL_DIR)
plc_manager = PLCManager()
_LAST_PLC_LOG = {"connected": None, "trigger": None, "ts": 0.0}


@app.route("/", methods=["GET"])
def home():
    models = model_manager.available()
    default_model = config.DEFAULT_MODEL if config.DEFAULT_MODEL in models else (models[0] if models else "")
    return render_template(
        "home.html",
        models=models,
        default_model=default_model,
    )


@app.route("/app", methods=["GET"])
def index():
    models = model_manager.available()
    default_model = config.DEFAULT_MODEL if config.DEFAULT_MODEL in models else (models[0] if models else "")
    return render_template(
        "index.html",
        models=models,
        classes=CLASS_META,
        default_model=default_model,
    )


@app.route("/plc/start", methods=["POST"])
def plc_start():
    ok = plc_manager.ensure_connected()
    return jsonify({"connected": ok, **plc_manager.status()})


@app.route("/plc/status", methods=["GET"])
def plc_status():
    if not plc_manager.connected:
        plc_manager.ensure_connected()
    status = plc_manager.status()
    connected = status.get("connected")
    trig = status.get("trigger")
    now = time.time()
    should_log = (
        connected != _LAST_PLC_LOG["connected"]
        or trig != _LAST_PLC_LOG["trigger"]
        or (now - _LAST_PLC_LOG["ts"] > 1.0)
    )
    if should_log:
        if connected:
            if trig == 1:
                print("[PLC] DB4.DBW0=1，等待截帧")
            else:
                print(f"[PLC] DB4.DBW0={trig}，未开始识别")
        else:
            print("[PLC] 未连接，无法读取 DB4.DBW0")
        _LAST_PLC_LOG.update({"connected": connected, "trigger": trig, "ts": now})
    return jsonify(status)


@app.route("/detect", methods=["POST"])
def detect():
    image: Optional[Image.Image] = None
    model_name = config.DEFAULT_MODEL
    plc_trigger_hint = 0

    if request.files:
        file = request.files.get("file")
        model_name = request.form.get("model") or model_name
        plc_trigger_hint = int(request.form.get("plc_trigger") or 0)
        if not file:
            return jsonify({"error": "缺少图像文件"}), 400
        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as exc:
            return jsonify({"error": f"图像解析失败: {exc}"}), 400
    else:
        payload = request.get_json(force=True, silent=True) or {}
        image_b64 = payload.get("image")
        model_name = payload.get("model") or model_name
        plc_trigger_hint = int(payload.get("plc_trigger") or 0)
        if not image_b64:
            return jsonify({"error": "缺少图像数据"}), 400
        try:
            image = _decode_image(image_b64)
        except Exception as exc:
            return jsonify({"error": f"图像解析失败: {exc}"}), 400

    if model_name not in model_manager.available():
        return jsonify({"error": f"模型未找到 {model_name}"}), 400

    # 未启/掉线时尝试自动重连一次，避免只能手动触发
    if not plc_manager.connected:
        plc_manager.ensure_connected()

    plc_status_val = None
    if plc_manager.connected:
        if plc_trigger_hint == 1 or plc_manager.trigger_recent():
            plc_status_val = 1
            print("[PLC] 前端已确认 DB4.DBW0=1 或最近已触发，跳过重复读取")
        else:
            plc_status_val = plc_manager.read_word(0)
            if plc_status_val is None:
                return jsonify({"error": "PLC 通信异常"}), 500
            if plc_status_val != 1:
                print(f"[PLC] DB4.DBW0={plc_status_val}，未开始识别（阻止本次推理）")
                return jsonify({"error": "PLC 未触发（DB4.DBW0 != 1）", "plc_status": plc_status_val}), 409
            print("[PLC] DB4.DBW0=1，开始识别当前帧")
    else:
        print("[PLC] 未连接，无法读取 DB4.DBW0")

    detections = model_manager.predict(image, model_name)
    counts: Dict[int, int] = {}
    best_det: Optional[Detection] = None
    for det in detections:
        counts[det.cls_id] = counts.get(det.cls_id, 0) + 1
        if best_det is None or det.conf > best_det.conf:
            best_det = det
    if best_det is None:
        # 与 main_pro2 一致：未检测到则归类为成品（ID=2）
        best_cls_id = 2
        counts[best_cls_id] = counts.get(best_cls_id, 0) + 1
        print("[识别] 未检测到目标，默认归类为成品(ID=2)")
    else:
        best_cls_id = best_det.cls_id
        print(f"[识别] 检出最优类别 ID={best_cls_id}")
    total = sum(counts.values())

    # 写回 PLC（DBW2=类别值，DBW0=2），先确保连接
    plc_connected = plc_manager.ensure_connected()
    plc_value = int(best_cls_id)  # model.py 已将 YOLO cls 变为 1-based
    if plc_connected:
        ok = plc_manager.write_result(plc_value)
        if ok:
            print(f"[PLC写入] 已写 DB4.DBW0=2, DB4.DBW2={plc_value}")
        else:
            print(f"[PLC写入] 失败：last_error={plc_manager.last_error}")
    else:
        print(f"[PLC写入] 未连接，无法写入 DB4.DBW2={plc_value}/DBW0=2")

    return jsonify(
        {
            "model": model_name,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "total": total,
            "counts": counts,
            "best_cls": best_cls_id,
            "plc": plc_manager.status(refresh_trigger=False),
            "plc_trigger": plc_status_val,
            "detections": [
                {
                    "cls_id": det.cls_id,
                    "cls_name": det.cls_name,
                    "conf": det.conf,
                    "box": det.box,
                }
                for det in detections
            ],
        }
    )


def _decode_image(image_b64: str) -> Image.Image:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
