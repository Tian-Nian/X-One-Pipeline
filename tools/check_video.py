import h5py
from typing import *
import cv2
import numpy as np
import os
import fnmatch
import sys
import select
from typing import Dict, Any, List
from skimage.metrics import structural_similarity as ssim
import argparse

def calculate_metrics(img1, img2):
    def mse(img1, img2):
        return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    result = {}
    result["PSNR"] = cv2.PSNR(img1, img2)
    result["MSE"] = mse(img1, img2)
    result["SSIM"] = ssim(img1, img2, channel_axis=-1, data_range=255)
    return result

def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=True)

if __name__ == "__main__":
    origin_npy = "data/raw_new/x-one/0_cam_head.npy"
    frames = load_npy(origin_npy)
    check_list = ["h264_libx_14", "h264_nv_14", "jpeg"]
    check_files = [f"data/{check_file}/x-one/0_cam_head.npy" for check_file in check_list]
    for check_file in check_files:
        check_frames = load_npy(check_file)
        all_psnr, all_mse, all_ssim = [], [], []
        for i in range(len(frames)):
            metrics = calculate_metrics(frames[i], check_frames[i])
            all_psnr.append(metrics["PSNR"])
            all_mse.append(metrics["MSE"])
            all_ssim.append(metrics["SSIM"])
        print(f"Check file: {check_file}")
        print(f"Average PSNR: {np.mean(all_psnr):.2f} dB")
        print(f"Average MSE: {np.mean(all_mse):.2f}")
        print(f"Average SSIM: {np.mean(all_ssim):.6f}")
    # print(f"Loaded {len(frames)} frames. Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")