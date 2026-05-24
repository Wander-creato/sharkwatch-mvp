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

        print(f"[SYSTEM] Connecting to YouTube live stream: {self.video_path}")
        # Resolve YouTube URL to a standard readable video stream (480p targeted for performance)
        try:
            self.cap = cap_from_youtube(self.video_path, "480p")
        except Exception as e:
            print(f"[HARDWARE ERROR] Failed to load YouTube stream: {e}")
            self.cap = cv2.VideoCapture(0)  # Fallback to webcam if stream resolution fails

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
            ret, frame = self.cap.read()
            if not ret:
                # Re-establish stream connection seamlessly if it hits the end or drops
                try:
                    self.cap = cap_from_youtube(self.video_path, "480p")
                except Exception:
                    time.sleep(2)
                continue
            with self.read_lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.03)  # Standardize feed to roughly ~30 FPS

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.cap.release()
