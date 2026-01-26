import sys
sys.path.append("./")

import numpy as np
import time

from robot.robot.base_robot import Robot
from robot.controller.Y1_controller import Y1Controller
from robot.sensor.Realsense_sensor import RealsenseSensor
from robot.data.collect_any import CollectAny

# setting your realsense serial
CAMERA_SERIALS = {
    'head': '233522079294',  # Replace with actual serial number
    #'left_wrist': '941322071558',   # Replace with actual serial number
     # 'left_wrist': '233722072561',
     'left_wrist': '233722072561',
    #'right_wrist': '337322073597',   # Replace with actual serial number
    # 'right_wrist': '233622078943',   # Replace with actual serial number
    'right_wrist': '337322073597',   # Replace with actual serial number
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
    "task_name": "deploy_pick_dual_bottles",
    "save_format": "hdf5",
    "save_freq": 30, 
}

class Y1Dual(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.controllers = {
            "arm":{
                "left_arm": Y1Controller("left_arm"),
                "right_arm": Y1Controller("right_arm"),
            }
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_left_wrist": RealsenseSensor("cam_left_wrist"),
                "cam_right_wrist": RealsenseSensor("cam_right_wrist"),
            },
        }

    def reset(self):
        # self.change_mode(teleop=False)
        time.sleep(1)

        move_data = {
            "arm":{
                "left_arm":{
                    # "joint": [0, 0, 0.3, 0.2, 0, 0],
                    "joint": [0, 0, 0, 0, 0, 0],
                    # "joint": [0.3, 0, 0, 0, 0, 0],
                    "gripper":  1.0,
                },
                
                "right_arm":{
                    # "joint": [0, 0, 0.3, 0.2, 0, 0],
                    "joint": [0, 0, 0, 0, 0, 0],
                    "gripper":  1.0,
                }
            }
        }
        self.move(move_data)
        time.sleep(2)
        # self.change_mode(teleop=True)
    
    def change_mode(self, teleop):
        if self.is_teleop == teleop:
            return
        else:
            time.sleep(1)
            self.controllers["arm"]["left_arm"].change_mode(teleop)
            self.controllers["arm"]["right_arm"].change_mode(teleop)
            time.sleep(1)
            self.is_teleop = teleop

    def set_up(self, teleop=False):
        self.controllers["arm"]["left_arm"].set_up("can1", teleop=teleop)
        self.controllers["arm"]["right_arm"].set_up("can0", teleop=teleop)

        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS['head'], is_depth=False)
        self.sensors["image"]["cam_left_wrist"].set_up(CAMERA_SERIALS['left_wrist'], is_depth=False)
        self.sensors["image"]["cam_right_wrist"].set_up(CAMERA_SERIALS['right_wrist'], is_depth=False)

        self.is_teleop = teleop
        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"]
                               })
    
    

if __name__ == "__main__":
    import time
    import cv2
    
    robot = Y1Dual()
    # robot.set_up(teleop=True)

    robot.set_up(teleop=False)
    # robot.reset()
    # robot.get()
    # robot.get()
    # robot.get()
    # robot.replay(data_path="./save/deploy_pick_dual_bottles/7.hdf5",key_banned=["qpos"])
    # time.sleep(2)
    # robot.show_pic("/mnt/nas/y1_real_data/nianhuo/0.hdf5", "cam_head")
    # data = robot.get()[1]["cam_head"]["color"]
    # cv2.imwrite("head.jpg", data)
    # robot.replay("./save/test/0.hdf5", key_banned=["qpos"])
    exit()

    # collection test
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
