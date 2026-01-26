import os
import fcntl
import mmap
import select
import numpy as np
import v4l2
import time
import cv2

class V4L2Camera:
    def __init__(self, device="/dev/video0", width=640, height=480):
        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

        # 设置格式
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = width
        fmt.fmt.pix.height = height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YUYV
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt)

        # 请求 buffer
        req = v4l2.v4l2_requestbuffers()
        req.count = 4
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)

        self.buffers = []

        for i in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.type = req.type
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = i
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf)

            mm = mmap.mmap(
                self.fd, buf.length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED,
                offset=buf.m.offset
            )
            self.buffers.append(mm)

            fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        # 开始采集
        buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMON, buf_type)

        self.width = width
        self.height = height

        # 对齐 offset
        self.offset_ns = None
    
    def read(self, timeout=2.0):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None

        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP

        fcntl.ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf)

        # === 时间戳（秒 + 微秒）===
        cam_ns = int((buf.timestamp.secs + buf.timestamp.usecs * 1e-6) * 1e9)
        # # === 只在第一次计算 offset ===
        # if self.offset_ns is None:
        #     self.offset_ns = time.monotonic_ns() - cam_ns
        
        print(cam_ns)

        aligned_ts_ns = cam_ns # + self.offset_ns

        data = self.buffers[buf.index][:buf.bytesused]

        # YUYV → numpy
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.height, self.width, 2)
        )

        # 重新入队
        fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        return aligned_ts_ns, frame

# === 使用 ===
cam = V4L2Camera("/dev/video4")

while True:
    ret = cam.read()
    if ret is None:
        continue

    ts_ns, frame = ret
    # print(frame.shape)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    print(f"aligned timestamp(ns): {ts_ns}")
    print(f"monotonic now(ns): {time.monotonic_ns()}")
    print(f"delta(ms): {(time.monotonic_ns() - ts_ns)/1e6:.3f}")
    
    # cv2.imshow("frame", frame_bgr)
    # cv2.waitKey(10)