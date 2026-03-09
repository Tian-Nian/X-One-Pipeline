import h5py
import json
import numpy as np
import argparse

def print_hdf5_structure(name, obj):
    """
    递归打印 HDF5 对象的结构信息。
    """
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        
        # 尝试打印数据集的内容
        if obj.ndim == 0:  # 标量数据集
            data = obj[()]
            if hasattr(data, 'decode'):
                data = data.decode('utf-8')
            print(f"{indent}  Value: {data}")
        else:
            # 过滤图片或大数据集：如果 dtype 是字节串 (S) 或者单元素过大，或者总数据量极大，则跳过
            # JPEG 压缩图片通常以 'S' (字节串) 存储且 size 较大
            is_image_or_large = False
            if obj.dtype.kind in ['S', 'V']: # 字节串或原始数据
                is_image_or_large = True
            elif obj.size > 100000: # 经验值：如果元素超过 10w 个 (如高分视频帧)
                is_image_or_large = True

            if is_image_or_large:
                print(f"{indent}  [Content omitted: Large or Binary data (e.g., Image/Video)]")
            else:
                # 打印数据集的前几个元素 (最多5个)
                try:
                    if obj.size > 0:
                        # 如果是多维数组，取第一维的前5个
                        sample_size = min(5, obj.shape[0])
                        data_sample = obj[:sample_size]
                        print(f"{indent}  Data Sample (first {sample_size}):")
                        # 使用 np.array2string 控制打印格式
                        print(f"{indent}    {np.array2string(data_sample, precision=4, separator=', ', edgeitems=3)}")
                except Exception as e:
                    print(f"{indent}  Could not read data: {e}")

def read_hdf5_structure(hdf5_path):
    """
    读取并打印 HDF5 文件的完整结构。
    """
    print(f"\n--- 正在分析 HDF5 结构: {hdf5_path} ---\n")
    try:
        with h5py.File(hdf5_path, "r") as f:
            f.visititems(print_hdf5_structure)
    except Exception as e:
        print(f"读取 HDF5 文件时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看 HDF5 文件结构')
    parser.add_argument('path', type=str, nargs='?', 
                        default="/home/pc/Desktop/bytedance_new/open_bag/xone-cement_full/0.hdf5",
                        help='HDF5 文件路径')
    args = parser.parse_args()
    
    read_hdf5_structure(args.path)
