from robot.sensor.sensor import Sensor
import numpy as np
from robot.utils.base.data_handler import debug_print
import subprocess
import threading

class BaseVisionSensor(Sensor):
    def __init__(self, TEST=False):
        super().__init__()
        self.name = "vision_sensor"
        self.type = "vision_sensor"
        self.collect_info = None
        self.encode_rgb = False
        self.TEST = TEST

        # H.264 实时编码相关状态
        self.is_h264 = False
        self.h264_process = None
        self.h264_thread = None
        self.h264_buffer = [] # 用于存放编码完成的视频帧片断

    def start_h264_encoding(self, width, height, fps=30, use_nvenc=True):
        """
        初始化 FFmpeg 管道进行实时 H.264 编码
        """
        if use_nvenc:
            cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", 
                "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "zerolatency",
                "-rc", "vbr", "-cq", "19", "-pix_fmt", "yuv420p",
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "pipe:1"
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", 
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-crf", "19", "-pix_fmt", "yuv420p",
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "pipe:1"
            ]

        self.h264_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        def read_output():
            while self.h264_process and self.h264_process.poll() is None:
                try:
                    chunk = self.h264_process.stdout.read(8192)
                    if chunk:
                        self.h264_buffer.append(chunk)
                    else:
                        break
                except Exception:
                    break

        def read_stderr():
            while self.h264_process and self.h264_process.poll() is None:
                try:
                    line = self.h264_process.stderr.readline()
                    if line:
                        debug_print(self.name, f"FFmpeg: {line.decode().strip()}", "DEBUG")
                    else:
                        break
                except Exception:
                    break

        self.h264_thread = threading.Thread(target=read_output, daemon=True)
        self.h264_thread.start()
        
        self.h264_err_thread = threading.Thread(target=read_stderr, daemon=True)
        self.h264_err_thread.start()

    def get_information(self):
        image_info = {}
        try:
            image = self.get_image()
        except Exception as e:
            debug_print(self.name, f"Pipe break: {e}", "ERROR")
            image = {}
            image["color"] = None
            image["depth"] = None
        
        if "color" in self.collect_info:
            if getattr(self, "is_jpeg", False):
                import cv2
                img_raw = image["color"]
                if img_raw is not None:
                    success, encoded_image = cv2.imencode('.jpg', image["color"]) # , [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    jpeg_data = encoded_image.tobytes()
                    image["color"] = jpeg_data
                    if self.TEST:
                        from robot.utils.base.data_handler import jpeg_test

                        result = jpeg_test(img_raw, jpeg_data)
                        print(f"{self.name} PSNR:", result["PSNR"])
                        print(f"{self.name} MSE:", result["MSE"])
                        print(f"{self.name} SSIM:", result["SSIM"])
            
            image_info["color"] = image["color"]
        if "depth" in self.collect_info:
            image_info["depth"] = image["depth"]
        if "point_cloud" in self.collect_info:
            image_info["point_cloud"] = image["point_cloud"]
        
        if "timestamp" in image.keys():
            image_info["timestamp"] = image["timestamp"]
        
        return image_info