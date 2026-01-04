import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_ROOT = BASE_DIR
MODEL_DIR = APP_ROOT / "model"
TEMPLATE_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "images"

DEFAULT_MODEL = os.getenv("YOLO_MODEL", "best.pt")
CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.1"))

# PLC settings (DB4: DBW0 trigger, DBW2 result)
PLC_IP = os.getenv("PLC_IP", "192.168.1.10")
PLC_DB = int(os.getenv("PLC_DB", "4"))
PLC_RACK = int(os.getenv("PLC_RACK", "0"))
PLC_SLOT = int(os.getenv("PLC_SLOT", "1"))
PLC_CONN_TYPE = int(os.getenv("PLC_CONN_TYPE", "2"))

FALLBACK_NAMES = ["病斑", "成品", "带泥", "分叉", "磕疤", "烂头", "锈", "芽孢"]
FALLBACK_COLORS = [
    "#ffca2f",
    "#ff6b7f",
    "#ff3b53",
    "#43f2ff",
    "#9ad8ff",
    "#3bd698",
    "#ff9f3f",
    "#e05bff",
]
NAME_TRANSLATIONS = {
    "bingban": "病斑",
    "chengpin": "成品",
    "daini": "带泥",
    "fencha": "分叉",
    "keba": "磕疤",
    "lantou": "烂头",
    "xiu": "锈",
    "yabao": "芽孢",
}
