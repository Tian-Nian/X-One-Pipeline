import h5py, json, cv2, argparse, subprocess, os
import numpy as np

def load_xspark_data(hdf5_path, decode_images=True):
    def decode_h264_stream(video_bytes):
        """
        使用 FFmpeg 从内存字节流中解码 H.264 帧
        """
        if not video_bytes:
            return np.array([])
        
        # 将字节转为 numpy 以获取 bytes（如果是 np.void 类型）
        cmd = [
            "ffmpeg", "-i", "pipe:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"
        ]
        
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        stdout_data, stderr_data = process.communicate(input=video_bytes.tobytes() if hasattr(video_bytes, 'tobytes') else video_bytes)
        
        if process.returncode != 0:
            print(f"FFmpeg decode error: {stderr_data.decode()}")
            return np.array([])

        # 由于每帧大小不确定，我们需要先搞清楚分辨率
        # 或者从 HDF5 的 shape dataset 中读取 (假设已经存了 shape)
        # 这里为了通用，尝试从 stderr 匹配分辨率（ffmpeg 默认会输出）
        import re
        match = re.search(r'Stream #.*: Video:.* (\d+)x(\d+)', stderr_data.decode())
        if match:
            w, h = map(int, match.groups())
            frames = np.frombuffer(stdout_data, dtype=np.uint8).reshape(-1, h, w, 3)
            return frames
        return np.array([])

    def decode_image(img_bytes):
        try:
            if isinstance(img_bytes, (bytes, np.bytes_)):
                jpeg_bytes = img_bytes.rstrip(b"\0")
            elif isinstance(img_bytes, np.ndarray) and img_bytes.dtype.kind in ['S', 'U']:
                jpeg_bytes = img_bytes.item().rstrip(b"\0")
            else:
                return img_bytes
            
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return img_bytes

    def h5_to_dict(obj):
        d = {}
        for key, item in obj.items():
            if isinstance(item, h5py.Group):
                d[key] = h5_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                val = item[()]
                
                if key == "colors" and isinstance(val, np.ndarray):
                    decoded_frames = []
                    for frame in val:
                        decoded_frames.append(frame if not decode_images else decode_image(frame))
                    d[key] = np.array(decoded_frames)
                    continue
                elif key=="video_h264":
                    # 解码 H.264 视频流
                    if decode_images:
                        # 尝试从同级组中获取 shape 以加速解码（如果有的话）
                        d[key] = decode_h264_stream(val)
                    else:
                        d[key] = val
                    continue

                if isinstance(val, (bytes, np.bytes_, np.ndarray)) and (val.dtype.kind in ['S', 'U']):
                    try:
                        if isinstance(val, np.ndarray) and val.size == 1:
                            val_item = val.item()
                        else:
                            val_item = val
                        
                        decoded_str = val_item.decode("utf-8")
                        try:
                            d[key] = json.loads(decoded_str)
                        except json.JSONDecodeError:
                            d[key] = decoded_str
                    except Exception:
                        d[key] = val
                else:
                    d[key] = val
        return d

    with h5py.File(hdf5_path, "r") as f:
        return h5_to_dict(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='HDF5 File Path')
    args = parser.parse_args()

    decoded_data_dict = load_xspark_data(args.path, decode_images=True)
    # breakpoint()
    cam_head_frames = decoded_data_dict["vision"]["cam_head"]["colors"]
    
    np.save(f"{os.path.splitext(args.path)[0]}_cam_head.npy", cam_head_frames)