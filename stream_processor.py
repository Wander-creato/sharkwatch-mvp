import cv2
import threading
import time
from cap_from_youtube import cap_from_youtube


class VideoStreamProcessor:
    def __init__(self, video_path="https://www.youtube.com/watch?v=p4yQ9qpTh1c"):
        self.video_path = video_path
        self.ret = False
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None
        self.cap = None
        self.last_error = None
        self.last_reconnect_attempt = 0
        self.youtube_resolutions = ("480p", "720p", "360p", "240p", "144p")

        print(f"[SYSTEM] Connecting to YouTube live stream: {self.video_path}")
        self.cap = self._open_capture()

    def _open_capture(self):
        capture = self._open_with_cap_from_youtube()
        if capture is not None:
            return capture

        capture = self._open_with_ytdlp()
        if capture is not None:
            return capture

        print("[SYSTEM] Falling back to local webcam capture.")
        return cv2.VideoCapture(0)

    def _open_with_cap_from_youtube(self):
        # Resolve YouTube URL to a standard readable video stream, trying common adaptive profiles.
        for resolution in self.youtube_resolutions:
            try:
                print(f"[SYSTEM] Attempting YouTube stream at {resolution}...")
                capture = cap_from_youtube(self.video_path, resolution)
                if capture is not None and capture.isOpened():
                    self.last_error = None
                    print(f"[SYSTEM] YouTube stream attached at {resolution}.")
                    return capture
                if capture is not None:
                    capture.release()
                self.last_error = f"YouTube stream at {resolution} did not open"
                print(f"[STREAM WARNING] {self.last_error}")
            except Exception as e:
                self.last_error = str(e)
                print(f"[STREAM WARNING] YouTube stream at {resolution} unavailable: {e}")
        return None

    def _open_with_ytdlp(self):
        try:
            import yt_dlp

            print("[SYSTEM] Resolving YouTube stream URL through yt-dlp fallback...")
            ydl_opts = {
                "format": "best[height<=480][ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]/best",
                "quiet": True,
                "no_warnings": False,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.video_path, download=False)

            stream_url = info.get("url")
            if not stream_url:
                formats = info.get("formats", [])
                for fmt in reversed(formats):
                    height = fmt.get("height") or 0
                    candidate_url = fmt.get("url")
                    if candidate_url and height <= 720:
                        stream_url = candidate_url
                        break

            if not stream_url:
                raise RuntimeError("yt-dlp did not return a playable stream URL")

            capture = cv2.VideoCapture(stream_url)
            if capture.isOpened():
                self.last_error = None
                print("[SYSTEM] YouTube stream attached through yt-dlp fallback.")
                return capture

            capture.release()
            raise RuntimeError("OpenCV could not open yt-dlp stream URL")
        except Exception as e:
            self.last_error = str(e)
            print(f"[STREAM WARNING] yt-dlp fallback failed: {e}")
            return None

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._reconnect()
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Re-establish stream connection seamlessly if it hits the end or drops
                    self._reconnect()
                    time.sleep(1)
                    continue

                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
                time.sleep(0.03)  # Standardize feed to roughly ~30 FPS
            except Exception as e:
                self.last_error = str(e)
                print(f"[STREAM ERROR] Frame acquisition fault: {e}")
                self._reconnect()
                time.sleep(2)

    def _reconnect(self):
        current_time = time.time()
        if current_time - self.last_reconnect_attempt < 2:
            return

        self.last_reconnect_attempt = current_time
        if self.cap is not None:
            self.cap.release()

        print("[SYSTEM] Reconnecting to video stream...")
        self.cap = self._open_capture()

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def status(self):
        with self.read_lock:
            has_frame = self.frame is not None
        return {
            "connected": bool(self.cap is not None and self.cap.isOpened()),
            "has_frame": has_frame,
            "last_error": self.last_error,
        }

    def stop(self):
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.cap is not None:
            self.cap.release()
