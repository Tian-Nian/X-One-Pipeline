import json
import numpy as np

import h5py
import cv2
import os

"""
# data/task_name/task_config/episode_00000.hdf5
vision:
    cam_head:
        colors:
        depths:    
        intrinsic_matrix:
        extrinsics_matrix:
        shape:
    cam_left_wrist:
    cam_right_wrist:
    (单臂)cam_wrist:
    (optional)cam_third_view:
 state:
     left_arm_joint_states:
     left_ee_joint_states: # 末端执行器关节状态（比如手的关节角）
     left_ee_poses: # 世界坐标系pose（xyz,qw,qx,qy,qz）
     left_tcp_poses:
     left_delta_ee_poses:   
     right_arm_joint_states:
     right_ee_joint_states:
     right_ee_poses:
     right_tcp_poses:
     right_delta_ee_poses:
     (单臂)arm_joint_states:
     (单臂)ee_joint_states:
     (单臂)ee_poses:
     (单臂)tcp_poses:
     (单臂)delta_ee_poses:
     mobile:
instructions: ["a", "b", "c"] # 任务级别instruciton
subtasks:
    [
        [(0, 12), "stage 1"], 
        [(12, 14), "stage 2"], 
    ]
    
additional_info:
    frequency: 
data_format_version: v1.0
"""
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_mp4(mp4_path):
    cap = cv2.VideoCapture(mp4_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV 默认是 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)

    cap.release()

    video_np = np.array(frames)
    return video_np

def load_x_ego_data(data_path, decode_images=False):
    hand_pose = load_json(os.path.join(data_path, "hand_poses.json"))
    head_camera = load_mp4(os.path.join(data_path, "cam_head_640_360.mp4")) # np.array of shape (T, H, W, 3) in BGR format
    left_wrist_camera = load_mp4(os.path.join(data_path, "cam_left_wrist_640_360.mp4"))
    right_wrist_camera = load_mp4(os.path.join(data_path, "cam_right_wrist_640_360.mp4"))
    raw_sub_tasks = load_json(os.path.join(data_path, "caption.json"))
    
    sub_tasks = []
    for raw_sub_task in raw_sub_tasks["atomic_action"]:
        raw_sub_task["frame_interval"][1] -= 1 # 将结束帧减1，使其变为左闭右闭区间，方便后续使用
        sub_task = (raw_sub_task["frame_interval"], raw_sub_task["instruction"])
        sub_tasks.append(sub_task)
    
    additional_info = {
        "frequency": cv2.VideoCapture(os.path.join(data_path, "cam_head_640_360.mp4")).get(cv2.CAP_PROP_FPS),
        "hand_betas": {
            "left_hand": hand_pose["left_hand"]["hand_betas"],
            "right_hand": hand_pose["right_hand"]["hand_betas"],
        },
        "init_wrist_global_pose":{
            "left_hand": np.concatenate([hand_pose["left_hand"]["init_wrist_trans_global"], hand_pose["left_hand"]["init_wrist_rot_global"]], axis=0),
            "right_hand": np.concatenate([hand_pose["right_hand"]["init_wrist_trans_global"], hand_pose["right_hand"]["init_wrist_rot_global"]], axis=0),
        }
    }
    data_format_version = "v1.0"


    data_dict = {
        "hand_pose": hand_pose,
        "head_camera": head_camera,
        "left_wrist_camera": left_wrist_camera,
        "right_wrist_camera": right_wrist_camera,
        "sub_task": sub_tasks,
        "instructions": [raw_sub_tasks["instruction"]],
        "additional_info": additional_info,
        "data_format_version": data_format_version,
    }
    return data_dict

def transform_x_ego_to_xspark(data_dict):
    # 这里需要根据 X-Spark 的数据格式进行转换
    # 例如，X-Spark 可能需要将视频帧解码为图像数组，或者将手部姿态数据转换为特定的格式
    transformed_data = {
        "vision": {
            "cam_head": {
                "colors": data_dict["head_camera"],
                "shape": data_dict["head_camera"][0].shape,
            },
            "cam_left_wrist": {
                "colors": data_dict["left_wrist_camera"],
                "shape": data_dict["left_wrist_camera"][0].shape,
            },
            "cam_right_wrist": {
                "colors": data_dict["right_wrist_camera"],
                "shape": data_dict["right_wrist_camera"][0].shape,
            }
        },
        "state": {
            "left_ee_joint_states": data_dict["hand_pose"]["left_hand"]["finger_state"],
            "right_ee_joint_states": data_dict["hand_pose"]["right_hand"]["finger_state"],
            "left_delta_ee_poses": np.concatenate([data_dict["hand_pose"]["left_hand"]["delta_wrist_trans"], \
                                                   data_dict["hand_pose"]["left_hand"]["delta_wrist_rot"] ], axis=1),
            "right_delta_ee_poses": np.concatenate([data_dict["hand_pose"]["right_hand"]["delta_wrist_trans"], \
                                                    data_dict["hand_pose"]["right_hand"]["delta_wrist_rot"] ], axis=1),
        },
        "sub_task": data_dict["sub_task"],
        "instructions": data_dict.get("instructions", []),
        "additional_info": data_dict.get("additional_info", {}),
        "data_format_version": data_dict.get("data_format_version", "v1.0"),
    }

    return transformed_data

def save_xspark(data, save_path):
    with h5py.File(save_path, "w") as f:
        vision = f.create_group("vision")
        state = f.create_group("state")
        cam_head = vision.create_group("cam_head")
        cam_head.create_dataset("colors", data=data["vision"]["cam_head"]["colors"])
        cam_head.create_dataset("shape", data=data["vision"]["cam_head"]["shape"])  

        cam_left_wrist = vision.create_group("cam_left_wrist")
        cam_left_wrist.create_dataset("colors", data=data["vision"]["cam_left_wrist"]["colors"])
        cam_left_wrist.create_dataset("shape", data=data["vision"]["cam_left_wrist"]["shape"]) # 固定分辨率

        cam_right_wrist = vision.create_group("cam_right_wrist")
        cam_right_wrist.create_dataset("colors", data=data["vision"]["cam_right_wrist"]["colors"])
        cam_right_wrist.create_dataset("shape", data=data["vision"]["cam_right_wrist"]["shape"]) # 固定分辨率
        
        state.create_dataset("left_ee_joint_states", data=data["state"]["left_ee_joint_states"])
        state.create_dataset("left_delta_ee_poses", data=data["state"]["left_delta_ee_poses"])
        state.create_dataset("right_ee_joint_states", data=data["state"]["right_ee_joint_states"])
        state.create_dataset("right_delta_ee_poses", data=data["state"]["right_delta_ee_poses"])

        f.create_dataset("instructions", data=np.string_(json.dumps(data["instructions"])))
        f.create_dataset("subtasks", data=np.string_(json.dumps(data["sub_task"])))
        addition_info = f.create_group("additional_info")
        # addition_info.create_dataset("frequency", data=data["additional_info"].get("frequency", 30))
        for key, value in data["additional_info"].items():
            if isinstance(value, dict):
                sub_group = addition_info.create_group(key)
                for sub_key, sub_value in value.items():
                    sub_group.create_dataset(sub_key, data=sub_value)
            else:
                addition_info.create_dataset(key, data=value)
        
        f.create_dataset("data_format_version", data=np.string_(data.get("data_format_version", "v1.0")))

if __name__ == "__main__":
    data_path = "/home/xspark-ai/project/merge/X-One-NT/data/episode000067"
    data_dict = load_x_ego_data(data_path)
    # breakpoint()

    data = transform_x_ego_to_xspark(data_dict)
    save_path = "/home/xspark-ai/project/merge/X-One-NT/data/episode000067/episode_00000.hdf5"
    save_xspark(data, save_path)
