"""
ORCA EYE — Stage 1: Camera / Video Source
==========================================
CameraSource supports:
  - Local webcam (int index)
  - Video file (str path)
  - IP Webcam MJPEG stream (http://...) via two strategies:
      Strategy A: cv2.VideoCapture(url)  — works for most streams
      Strategy B: JPEG polling via /shot.jpg — robust fallback for Android
        IP Webcam app whose MJPEG boundary format confuses OpenCV/FFmpeg

Inputs  : source (int for webcam, str for file/URL), config dict
Outputs : (success, frame, timestamp) per read()
"""

import io
import logging
import time
import threading
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MJPEG Thread-Safe Frame Buffer
# Used by _MJPEGStreamReader to decouple network I/O from the main loop
# ---------------------------------------------------------------------------

class _FrameBuffer:
    """Thread-safe single-frame buffer."""

    def __init__(self):
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = time.time()
        self._new_frame.set()

    def get(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._frame, self._timestamp

    def wait(self, timeout: float = 1.0) -> bool:
        result = self._new_frame.wait(timeout)
        self._new_frame.clear()
        return result


# ---------------------------------------------------------------------------
# JPEG Polling Reader (Fallback for Android IP Webcam)
# ---------------------------------------------------------------------------

class _JPEGPollingReader:
    """
    Polls /shot.jpg at the target FPS.
    Used when cv2.VideoCapture cannot parse the MJPEG multipart boundary.

    Compatible with: IP Webcam (Android), DroidCam, most HTTP JPEG servers.
    """

    def __init__(self, base_url: str, fps_target: int = 15) -> None:
        # Derive the shot.jpg URL from the base
        self.shot_url = self._build_shot_url(base_url)
        self.interval = 1.0 / max(fps_target, 1)
        self._buf = _FrameBuffer()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="orca_jpeg_poll"
        )
        self._ok = False

        logger.info("JPEG polling fallback: %s @ %d fps", self.shot_url, fps_target)

    def start(self) -> bool:
        """Start polling thread. Returns True if first frame received."""
        self._thread.start()
        return self._buf.wait(timeout=5.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        frame, ts = self._buf.get()
        if frame is None:
            return False, None, time.time()
        return True, frame.copy(), ts

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        try:
            import requests
        except ImportError:
            logger.error("'requests' not installed. Run: pip install requests")
            return

        session = requests.Session()
        session.timeout = 3.0

        while not self._stop.is_set():
            t0 = time.time()
            try:
                resp = session.get(self.shot_url, timeout=3.0)
                if resp.status_code == 200 and resp.content:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._buf.put(frame)
            except Exception as exc:
                logger.debug("JPEG poll error: %s", exc)

            elapsed = time.time() - t0
            sleep = max(0.0, self.interval - elapsed)
            time.sleep(sleep)

    @staticmethod
    def _build_shot_url(url: str) -> str:
        """Build the /shot.jpg snapshot URL from any IP Webcam URL."""
        # Strip trailing path and append /shot.jpg
        base = url.split("//", 1)
        if len(base) < 2:
            return url
        host_and_path = base[1].split("/", 1)[0]  # just host:port
        return f"http://{host_and_path}/shot.jpg"


# ---------------------------------------------------------------------------
# MJPEG Stream Reader (Primary for IP streams)
# ---------------------------------------------------------------------------

class _MJPEGStreamReader:
    """
    Reads MJPEG frames directly from a multipart HTTP stream using requests.
    Handles the --myboundary format used by Android IP Webcam.
    Runs in a background thread; exposes read() for the main loop.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._buf = _FrameBuffer()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="orca_mjpeg_reader"
        )

    def start(self) -> bool:
        self._thread.start()
        return self._buf.wait(timeout=8.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        frame, ts = self._buf.get()
        if frame is None:
            return False, None, time.time()
        return True, frame, ts

    def stop(self) -> None:
        self._stop.set()

    def _stream_loop(self) -> None:
        try:
            import requests
        except ImportError:
            logger.error("'requests' not installed. Run: pip install requests")
            return

        session = requests.Session()
        while not self._stop.is_set():
            try:
                logger.info("Connecting to MJPEG stream: %s", self.url)
                with session.get(self.url, stream=True, timeout=10) as resp:
                    resp.raise_for_status()
                    buf = bytearray()
                    for chunk in resp.iter_content(chunk_size=65536):
                        if self._stop.is_set():
                            break
                        buf.extend(chunk)
                        while True:
                            start = buf.find(b'\xff\xd8')
                            if start == -1:
                                if len(buf) > 131072:
                                    del buf[:-4]
                                break
                            if start > 0:
                                del buf[:start]
                            end = buf.find(b'\xff\xd9', 2)
                            if end == -1:
                                break
                            jpg_bytes = bytes(buf[:end + 2])
                            del buf[:end + 2]
                            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                self._buf.put(frame)

            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("MJPEG stream error: %s — retrying in 2s", exc)
                    time.sleep(2.0)


# ---------------------------------------------------------------------------
# CameraSource — Unified Interface
# ---------------------------------------------------------------------------

class CameraSource:
    """
    Unified camera abstraction:
      - int  → cv2.VideoCapture(index)  (local webcam)
      - str path → cv2.VideoCapture(path)  (video file)
      - str http:// → tries MJPEG stream reader, falls back to JPEG polling

    All callers use the same read() → (success, frame, timestamp) interface.
    """

    def __init__(self, source: Union[int, str], cfg: dict) -> None:
        self.source = source
        self.cfg = cfg
        self.width: int = cfg.get("width", 640)
        self.height: int = cfg.get("height", 480)
        self.fps_target: int = cfg.get("fps_target", 30)
        self.reconnect_attempts: int = cfg.get("reconnect_attempts", 5)
        self.reconnect_delay: float = cfg.get("reconnect_delay_s", 1.0)

        self.is_file: bool = isinstance(source, str) and not source.startswith("http")
        self.is_ip_cam: bool = isinstance(source, str) and source.startswith("http")
        self.is_webcam: bool = isinstance(source, int)

        self.cap: Optional[cv2.VideoCapture] = None
        self._mjpeg_reader: Optional[_MJPEGStreamReader] = None
        self._jpeg_poller: Optional[_JPEGPollingReader] = None
        self._use_polling: bool = False
        self._use_mjpeg_reader: bool = False

        self.frame_count: int = 0
        self.total_frames: int = 0

        self._open()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read the next frame.
        Returns: (success, frame_bgr, timestamp)
        """
        frame: Optional[np.ndarray] = None
        ts = time.time()
        ok = False

        if self._use_mjpeg_reader and self._mjpeg_reader is not None:
            ok, frame, ts = self._mjpeg_reader.read()
            if not ok and self._jpeg_poller is not None:
                ok, frame, ts = self._jpeg_poller.read()

        elif self._use_polling and self._jpeg_poller is not None:
            ok, frame, ts = self._jpeg_poller.read()

        else:
            # Standard cv2.VideoCapture path
            if self.cap is None or not self.cap.isOpened():
                if not self._reconnect():
                    return False, None, time.time()

            ret, frame = self.cap.read()
            ts = time.time()
            if not ret:
                if self.is_file:
                    logger.info("End of video file reached.")
                    return False, None, ts
                logger.warning("Frame read failed, attempting reconnect.")
                if not self._reconnect():
                    return False, None, ts
                ret, frame = self.cap.read()
                ts = time.time()
                if not ret:
                    return False, None, ts
            ok = True

        if ok and frame is not None:
            self.frame_count += 1
            # Maintain configured resolution to ensure real-time GPU/CPU pipeline performance
            if self.width > 0 and self.height > 0:
                fh, fw = frame.shape[:2]
                if fw != self.width or fh != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            return True, frame, ts

        return False, None, time.time()

    def release(self) -> None:
        if self._mjpeg_reader:
            self._mjpeg_reader.stop()
        if self._jpeg_poller:
            self._jpeg_poller.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera source released.")

    def get_fps(self) -> float:
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS) or float(self.fps_target)
        return float(self.fps_target)

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self.is_ip_cam:
            self._open_ip_cam()
        else:
            self._open_cv2()

    def _open_ip_cam(self) -> None:
        """
        Try to connect to IP webcam using multiple strategies:
        1. Custom MJPEG stream reader (handles Android boundary format)
        2. JPEG polling fallback (/shot.jpg)
        3. Raw cv2.VideoCapture as last resort
        """
        logger.info("IP Webcam source: %s (target pipeline resolution: %dx%d)", self.source, self.width, self.height)

        # Strategy 1: Custom MJPEG reader (byte-level JPEG SOI/EOI parsing)
        logger.info("Trying MJPEG stream reader …")
        reader = _MJPEGStreamReader(self.source)
        if reader.start():
            logger.info("MJPEG stream reader connected successfully.")
            self._mjpeg_reader = reader
            self._use_mjpeg_reader = True
            logger.info("Pipeline operating resolution: %dx%d", self.width, self.height)
            return

        logger.warning("MJPEG reader failed. Trying JPEG polling fallback …")
        reader.stop()

        # Strategy 2: JPEG polling (/shot.jpg)
        poller = _JPEGPollingReader(self.source, fps_target=self.fps_target)
        if poller.start():
            logger.info("JPEG polling connected successfully.")
            self._jpeg_poller = poller
            self._use_polling = True
            logger.info("Pipeline operating resolution (polling): %dx%d", self.width, self.height)
            return

        logger.warning("JPEG polling failed. Trying raw cv2.VideoCapture …")
        poller.stop()

        # Strategy 3: cv2.VideoCapture
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            logger.info("cv2.VideoCapture connected: target %dx%d", self.width, self.height)
        else:
            raise RuntimeError(
                f"All three connection strategies failed for: {self.source}\n"
                "  Check:\n"
                "  1. Phone and PC are on the same Wi-Fi network\n"
                "  2. IP Webcam app is running and streaming\n"
                "  3. The URL is correct (open it in a browser first)"
            )

    def _open_cv2(self) -> None:
        logger.info("Opening source: %s", self.source)
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self.source!r}. "
                "Check device index or file path."
            )

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        else:
            orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Scale video to optimized operating resolution (max dimension 640)
            if orig_h > orig_w:
                target_h = 640
                target_w = int(orig_w * (640.0 / orig_h))
                target_w = target_w if target_w % 2 == 0 else target_w + 1
            else:
                target_w = 640
                target_h = int(orig_h * (640.0 / orig_w))
                target_h = target_h if target_h % 2 == 0 else target_h + 1

            self.width = target_w
            self.height = target_h
            logger.info(
                "Video file: %dx%d (scaled to %dx%d) @ %.1f fps, %d frames",
                orig_w, orig_h, self.width, self.height,
                self.cap.get(cv2.CAP_PROP_FPS), self.total_frames,
            )

        logger.info("Camera source opened: %s [%dx%d]", self.source, self.width, self.height)

    def _reconnect(self) -> bool:
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning("Reconnect attempt %d/%d …", attempt, self.reconnect_attempts)
            if self.cap is not None:
                self.cap.release()
            time.sleep(self.reconnect_delay)
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                logger.info("Reconnected successfully.")
                return True
        logger.error("All reconnect attempts failed.")
        return False

    def __repr__(self) -> str:
        mode = "ip_cam" if self.is_ip_cam else ("file" if self.is_file else "webcam")
        return f"CameraSource({mode}, {self.width}x{self.height}, frames={self.frame_count})"
