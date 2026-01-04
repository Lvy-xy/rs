import threading
import time
from typing import Dict, Optional

from . import config

try:
    import snap7  # type: ignore
except Exception:  # pragma: no cover
    snap7 = None

RETRY_DELAY_SEC = 0.02
FAST_TRIGGER_WINDOW_SEC = 0.1


class PLCManager:
    def __init__(self):
        self.ip = config.PLC_IP
        self.db = config.PLC_DB
        self.rack = config.PLC_RACK
        self.slot = config.PLC_SLOT
        self.conn_type = config.PLC_CONN_TYPE
        self.client: Optional["snap7.client.Client"] = None
        self.connected = False
        self.lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.exec_count = 0
        self.last_trigger: Optional[int] = None
        self.last_trigger_ts = 0.0
        self.last_result: Optional[int] = None

    def connect(self) -> bool:
        if snap7 is None:
            self.last_error = "snap7 未安装"
            self.connected = False
            return False
        # 若已连接则直接返回，避免重复建连
        if self.connected and self.client:
            return True
        with self.lock:
            self.client = snap7.client.Client()
            self.client.set_connection_type(self.conn_type)
            try:
                self.client.connect(self.ip, self.rack, self.slot)
                self.connected = self.client.get_connected()
                if not self.connected:
                    self.last_error = "无法连接 PLC"
            except Exception as exc:
                self.last_error = str(exc)
                self.connected = False
        return self.connected

    def disconnect(self):
        with self.lock:
            if self.client and self.client.get_connected():
                self.client.disconnect()
            self.connected = False

    def ensure_connected(self) -> bool:
        """保证连接可用；断开时自动重连一次。"""
        if self.connected and self.client:
            return True
        return self.connect()

    def read_word(self, offset: int) -> Optional[int]:
        with self.lock:
            if not (self.client and self.connected):
                return None
            try:
                data = self.client.db_read(self.db, offset, 2)
                value = int.from_bytes(data, byteorder="big", signed=True)
                if offset == 0:
                    self.last_trigger = value
                    self.last_trigger_ts = time.time()
                return value
            except Exception as exc:
                self.last_error = f"read DB{self.db}.DBW{offset} failed: {exc}"
                # 出错时标记断开，让前端/接口触发重连
                self.connected = False
                return None

    def write_word(self, offset: int, value: int, max_retries: int = 3) -> bool:
        value = max(-32768, min(32767, int(value)))
        for _ in range(max_retries):
            with self.lock:
                if not (self.client and self.connected):
                    return False
                try:
                    self.client.db_write(self.db, offset, value.to_bytes(2, byteorder="big", signed=True))
                    return True
                except Exception as exc:
                    self.last_error = f"write DB{self.db}.DBW{offset} failed: {exc}"
            time.sleep(RETRY_DELAY_SEC)
        # 多次失败后视为断开，便于外部重连
        self.connected = False
        return False

    def write_result(self, cls_id: int, max_retries: int = 3, confirm: bool = False) -> bool:
        """Single PLC round trip: DBW0=2 (done), DBW2=cls_id.

        Keep the write path单次往返，避免额外确认读阻塞（默认关闭确认以减小延迟）。
        """
        status_val = max(-32768, min(32767, 2))
        cls_val = max(-32768, min(32767, int(cls_id)))
        payload = status_val.to_bytes(2, byteorder="big", signed=True) + cls_val.to_bytes(
            2, byteorder="big", signed=True
        )
        for _ in range(max_retries):
            with self.lock:
                if not (self.client and self.connected):
                    return False
                try:
                    # write DBW0 and DBW2 together to cut latency
                    self.client.db_write(self.db, 0, payload)
                    self.exec_count += 1
                    self.last_result = cls_val
                    # 可选的单次轻量确认，不重试，避免增加等待
                    if confirm:
                        self._confirm_result(status_val, cls_val, retries=0)
                    return True
                except Exception as exc:
                    self.last_error = f"write_result failed: {exc}"
            time.sleep(RETRY_DELAY_SEC)
        self.connected = False
        return False

    def _confirm_result(self, status_val: int, cls_val: int, retries: int = 0) -> bool:
        if not (self.client and self.connected):
            return False
        for _ in range(max(0, retries) + 1):
            try:
                data = self.client.db_read(self.db, 0, 4)
            except Exception as exc:
                self.last_error = f"confirm DB{self.db} write failed: {exc}"
                self.connected = False
                return False

            status = int.from_bytes(data[:2], byteorder="big", signed=True)
            result = int.from_bytes(data[2:], byteorder="big", signed=True)
            if status == status_val and result == cls_val:
                self.last_trigger = status
                self.last_trigger_ts = time.time()
                return True
        return False

    def trigger_recent(self, window_sec: float = FAST_TRIGGER_WINDOW_SEC) -> bool:
        return self.last_trigger == 1 and (time.time() - self.last_trigger_ts) <= window_sec

    def status(self, refresh_trigger: bool = True) -> Dict:
        if self.connected:
            trigger = self.read_word(0) if refresh_trigger else self.last_trigger
        else:
            trigger = None
        return {
            "connected": bool(self.connected),
            "ip": self.ip,
            "db": self.db,
            "last_error": self.last_error,
            "exec_count": self.exec_count,
            "last_result": self.last_result,
            "trigger": trigger,
        }
