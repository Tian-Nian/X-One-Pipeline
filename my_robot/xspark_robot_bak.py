import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.Y1_controller import Y1Controller
from sensor.Cv_sensor import CvSensor
from sensor.Realsense_sensor import RealsenseSensor

from data.collect_any import CollectAny
# from utils.data_transofrm_pipeline import image_rgb_encode_pipeline, general_hdf5_rdt_format_pipeline
from utils.data_handler import debug_print, hdf5_groups_to_dict, load_hdf5_as_dict# , dict_to_list
import time

from scipy.spatial.transform import Rotation as R
import cv2

# setting your realsense serial
CAMERA_SERIALS = {
    'head': 0,  # Replace with actual serial number
    'left_wrist': 4,   # Replace with actual serial number
    'right_wrist': 2,   # Replace with actual serial number
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "save_path": "./save/",
    "task_name": "test",
    "save_format": "hdf5",
    "save_freq": 10, 
}

def pose_to_T(pose):
    """
    pose: [x, y, z, rx, ry, rz] (rpy or rotvec)
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_rotvec(pose[3:6]).as_matrix()
    return T

def T_to_pose(T):
    pose = np.zeros(6)
    pose[:3] = T[:3, 3]
    pose[3:6] = R.from_matrix(T[:3, :3]).as_rotvec()
    return pose

def se3_exp(delta):
    """
    delta: [dx, dy, dz, rx, ry, rz]  (twist)
    """
    delta = np.asarray(delta)
    v = delta[:3]
    w = delta[3:6]

    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(w).as_matrix()
    T[:3, 3] = v
    return T

def se3_delta(T_now, T_next):
    """
    return: 6D twist [dx, dy, dz, rx, ry, rz]
    """
    T_delta = np.linalg.inv(T_now) @ T_next

    # translation
    delta_p = T_delta[:3, 3]

    # rotation (log SO(3))
    delta_r = R.from_matrix(T_delta[:3, :3]).as_rotvec()

    return np.concatenate([delta_p, delta_r])

def compute_action(now_qpos, next_qpos):
    now_qpos = np.asarray(now_qpos)
    next_qpos = np.asarray(next_qpos)

    # left arm
    T_l_now  = pose_to_T(now_qpos[0:6])
    T_l_next = pose_to_T(next_qpos[0:6])
    l_eef_action = se3_delta(T_l_now, T_l_next)
    l_gripper = next_qpos[6:7]   # absolute

    # right arm
    T_r_now  = pose_to_T(now_qpos[7:13])
    T_r_next = pose_to_T(next_qpos[7:13])
    r_eef_action = se3_delta(T_r_now, T_r_next)
    r_gripper = next_qpos[13:14] # absolute

    action = np.concatenate([
        l_eef_action,
        l_gripper,
        r_eef_action,
        r_gripper
    ])

    return action

def apply_action(now_qpos, action):
    now_qpos = np.asarray(now_qpos)
    action = np.asarray(action)

    next_qpos = now_qpos.copy()

    # ---------- left arm ----------
    T_l_now = pose_to_T(now_qpos[:6])
    T_l_delta = se3_exp(action[:6])
    T_l_next = T_l_now @ T_l_delta

    next_qpos[0:6] = T_to_pose(T_l_next)
    # next_qpos[6] = action[6]   # gripper: absolute

    # ---------- right arm ----------
    # T_r_now = pose_to_T(now_qpos[7:13])
    # T_r_delta = se3_exp(action[7:13])
    # T_r_next = T_r_now @ T_r_delta

    # next_qpos[7:13] = T_to_pose(T_r_next)
    # next_qpos[13] = action[13] # gripper: absolute

    return next_qpos

class XsparkRobot(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.controllers = {
            "arm":{
                "left_arm": Y1Controller("left_arm"),
                "right_arm": Y1Controller("right_arm"),
            },
        }
        self.sensors = {
            "image": {
                "cam_head": CvSensor("cam_head"),
                "cam_left_wrist": CvSensor("cam_left_wrist"),
                "cam_right_wrist": CvSensor("cam_right_wrist"),
            },
        }
        # self.collection._add_data_transform_pipeline(image_rgb_encode_pipeline)

    def set_up(self, teleop=False):
        # super().set_up()

        self.controllers["arm"]["left_arm"].set_up("can1", teleop=teleop)
        self.controllers["arm"]["right_arm"].set_up("can0", teleop=teleop)

        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS['head'], is_depth=False)
        self.sensors["image"]["cam_left_wrist"].set_up(CAMERA_SERIALS['left_wrist'], is_depth=False)
        self.sensors["image"]["cam_right_wrist"].set_up(CAMERA_SERIALS['right_wrist'], is_depth=False)
        
        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                                "image": ["color"]
                               })
        print("set up success!")

    def reset(self):
        # self.change_mode(teleop=False)
        time.sleep(1)
        move_data = {
            "arm":{
                "left_arm":{
                    # "joint": [0, 0, 0.3, 0.2, 0, 0],
                    "joint": [0, 0, 0, 0, 0, 0],
                    # "qpos": [0.2, 0.0, 0.30, 0., 0., 0.],
                    "gripper":  1.0,
                },
                
                "right_arm":{
                    # "joint": [0, 0, 0.3, 0.2, 0, 0],
                    "joint": [0, 0, 0, 0, 0, 0],
                    # "qpos": [0.2, 0.0, 0.30, 0., 0., 0.],
                    "gripper":  1.0,
                }
            }
        }
        self.move(move_data)
        time.sleep(3)
        # self.change_mode(teleop=False)
    
    # ======================== EXTRA ======================== #
    def change_mode(self, teleop):
        self.controllers["arm"]["left_arm"].change_mode(teleop)
        self.controllers["arm"]["right_arm"].change_mode(teleop)
        time.sleep(1)
    
    def set_map(self, map_path):
        self.controllers["mobile"]["slamware"].set_map(map_path)

    def play_delta_qpos(self, data_path, scale=1.0):
        # episode = hdf5_groups_to_dict(data_path)
        # print(episode.keys())
        # import pdb;pdb.set_trace()
        ep_dict = hdf5_groups_to_dict(data_path)
        episode = []
        length = len(ep_dict["left_hand"]["delta_wrist_rot"])

        for i in range(length):
            ep = {}
            ep["left_arm"] = {
                "delta_wrist_trans": ep_dict["left_hand"]["delta_wrist_trans"][i],
                "delta_wrist_rot": ep_dict["left_hand"]["delta_wrist_rot"][i],
            }
            ep["right_arm"] = {
                "delta_wrist_trans": ep_dict["right_hand"]["delta_wrist_trans"][i],
                "delta_wrist_rot": ep_dict["right_hand"]["delta_wrist_rot"][i],
            }
            episode.append(ep)

        l_base_qpos, r_base_qpos = self.get()[0]["left_arm"]["qpos"], self.get()[0]["right_arm"]["qpos"] 
        for ep in episode:
            l_qpos, r_qpos = rot2euler(ep, scale)
            
            r_abs_qpos = apply_action(r_base_qpos, r_qpos)
            l_abs_qpos = apply_action(l_base_qpos, l_qpos)

            move_data = {
                "arm":{
                    "left_arm":{
                        "qpos": l_abs_qpos,
                    },
                    "right_arm":{
                        "qpos": r_abs_qpos,
                    }
                }  
            }
            self.move(move_data)

            time.sleep(0.1)

def rot2euler(ep, scale):
    l_eef, l_rot = ep["left_arm"]["delta_wrist_trans"] * scale, ep["left_arm"]["delta_wrist_rot"] 
    # 1. rotvec -> Rotation
    l_rot = R.from_rotvec(l_rot)

    # 2. Rotation -> Euler
    l_euler_xyz = l_rot.as_euler("xyz", degrees=False)

    l_qpos = np.concatenate([l_eef, l_euler_xyz])

    r_eef, r_rot = ep["right_arm"]["delta_wrist_trans"] * scale, ep["right_arm"]["delta_wrist_rot"] 
    # 1. rotvec -> Rotation
    r_rot = R.from_rotvec(r_rot)

    # 2. Rotation -> Euler
    r_euler_xyz = r_rot.as_euler("xyz", degrees=False)

    r_qpos = np.concatenate([r_eef, r_euler_xyz])
    return l_qpos, r_qpos


def read(data_path):
    episode = hdf5_groups_to_dict(data_path)
    print(episode.keys())

    # img = episode["cam_head"][0]

    # jpeg_bytes = np.array(img).tobytes().rstrip(b"\0")
    # nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    # img = cv2.imdecode(nparr, 1)
    # cv2.imwrite("cam_head.jpg", img)
            

if __name__ == "__main__":
    import time
    # read("/mnt/nas/demo0116/clips_hdf5_with_video/0.hdf5")
    robot = XsparkRobot()
    # robot.set_up(teleop=True)

    robot.set_up(teleop=False)
    robot.reset()
    # robot.get()
    robot.replay(data_path="save/90.hdf5", key_banned=["qpos"])
    # robot.play_delta_qpos(data_path="/mnt/nas/demo0116/clips_hdf5/0.hdf5", scale=0.8)
    # robot.show_pic("save/replay/0.hdf5", "cam_head")
    exit()