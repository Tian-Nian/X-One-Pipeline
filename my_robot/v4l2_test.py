import sys
sys.path.append("./")

import numpy as np

from robot.robot.base_robot import Robot

from robot.controller.Y1_controller import Y1Controller
# from robot.sensor.Cv_sensor import CvSensor
from robot.sensor.V4l2_sensor import V4l2Sensor

from robot.data.collect_any import CollectAny
from robot.utils.base.data_transform_pipeline import X_one_format_pipeline
from robot.utils.base.data_handler import is_enter_pressed
import time
import cv2
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

class XsparkRobot(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)
        self.first_start = True
        self.controllers = {
            "arm":{
                "left_arm": Y1Controller("left_arm"),
            },
        }
        self.sensors = {
            "image": {
                "cam_left_wrist": V4l2Sensor("cam_left_wrist"),
            },
        }

    def set_up(self, teleop=False):
        super().set_up()

        self.controllers["arm"]["left_arm"].set_up("can0", teleop=teleop)

        self.sensors["image"]["cam_left_wrist"].set_up("/dev/video4", is_depth=False, is_jepg=False)
        
        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                                "image": ["color"]
                               })
        print("set up success!")

    # ======================== EXTRA ======================== #
    def change_mode(self, teleop):
        self.controllers["arm"]["left_arm"].change_mode(teleop)
        time.sleep(1)
    
if __name__ == "__main__":
    import time
    
    robot = XsparkRobot(move_check=False)

    robot.test("save/test/0.hdf5")
    exit()
    robot.set_up(teleop=True)
    while True:
        data = robot.get()
        print(data[0]["left_arm"]["gripper"])
        cv2.imshow("img",data[1]["cam_left_wrist"]["color"])
        cv2.waitKey(1)