import cv2
import numpy as np
import time
import threading
from collections import deque
from robot.sensor.vision_sensor import VisionSensor
from robot.utils.base.data_handler import debug_print


class CvSensor(VisionSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.cap = None
        self.is_depth = False
        self.is_jepg = False
        self.is_undistort = False

        # 多线程控制
        self.frame_buffer = deque(maxlen=1)  # 保留最新一帧
        self.lock = threading.Lock()
        self.thread = None
        self.exit_event = threading.Event()
        self.keep_running = False

    def set_up(self, device_index='', start_event=None, is_depth=False, is_jepg=False, is_undistort=False):
        self.is_depth = is_depth
        self.is_jepg = is_jepg
        self.is_undistort = is_undistort
        self.start_event = start_event

        if self.is_undistort:
            self.calib = np.load(f"save/calibrate/{device_index}.npz")

        # 打开摄像头
        # try:
        #     self.cap = cv2.VideoCapture(f"/dev/{device_index}")
        # except:
        #     self.cap = cv2.VideoCapture(int(device_index))
        self.cap = cv2.VideoCapture(int(device_index))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {device_index}")

        # 设置分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # 启动线程采集
        self.keep_running = True
        self.exit_event.clear()
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()

        print(f"[{self.name}] Camera started (index={device_index})")

    def _update_frames(self):
        """独立线程不断读取摄像头帧"""
        try:
            if self.start_event is not None:
                while not self.exit_event.is_set():
                    time.sleep(0.0001)
                debug_print(self.name, "Get start event!", "INFO")
            
            while not self.exit_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    debug_print(self.name, "Failed to read frame", "ERROR")
                    time.sleep(0.01)
                    continue

                frame_data = {}

                if "color" in self.collect_info:
                    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.is_undistort:
                        color = self._undistort_fisheye(color)
                    if self.is_jepg:
                        success, encoded = cv2.imencode('.jpg', color)
                        color = encoded.tobytes()
                    frame_data["color"] = color

                if "depth" in self.collect_info:
                    if not self.is_depth:
                        debug_print(self.name, "Depth capture not enabled", "ERROR")
                        raise ValueError("Depth capture not enabled")
                    frame_data["depth"] = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)

                if frame_data:
                    frame_data["timestamp"] = time.time_ns()

                    with self.lock:
                        self.frame_buffer.append(frame_data)
        except Exception as e:
            debug_print(self.name, f"Thread exception: {e}", "ERROR")

    def get_image(self):
        """阻塞式获取最新帧"""
        while True:
            with self.lock:
                if self.frame_buffer:
                    return self.frame_buffer[-1]
            time.sleep(0.001)

    def _undistort_fisheye(self, img):
        K = self.calib["K"]
        D = self.calib["D"]
        h, w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    def cleanup(self):
        """释放摄像头和线程资源"""
        try:
            self.exit_event.set()
            self.keep_running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            debug_print(self.name, f"Cleanup error: {e}", "ERROR")

    def __del__(self):
        self.cleanup()