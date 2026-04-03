import sys
sys.path.append("./")

from robot.controller.arm_controller import ArmController
from robot.utils.base.data_handler import debug_print, is_enter_pressed

from pyAgxArm import create_agx_arm_config, AgxArmFactory
import numpy as np
import time

'''
Piper base code from:
https://github.com/agilexrobotics/pyAgxArm.git
'''

class PiperController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None

    def _collect_enable_diagnostics(self):
        arm_status = self.controller.get_arm_status()
        driver_states = [self.controller.get_driver_states(i) for i in range(1, 7)]

        return {
            "arm_ok": self.controller.is_ok(),
            "arm_fps": self.controller.get_fps(),
            "gripper_ok": self.end_effector.is_ok(),
            "gripper_fps": self.end_effector.get_fps(),
            "arm_status_ready": arm_status is not None,
            "driver_states_ready": [state is not None for state in driver_states],
            "joint_enable_status": self.controller.get_joints_enable_status_list(),
        }

    def _wait_until_enabled(self, timeout=3.0, interval=0.1):
        start_time = time.monotonic()
        last_diagnostics = self._collect_enable_diagnostics()

        while time.monotonic() - start_time < timeout:
            if self.controller.enable():
                return

            time.sleep(interval)
            last_diagnostics = self._collect_enable_diagnostics()

        raise RuntimeError(
            f"Failed to enable Piper on CAN channel {self.controller.get_channel()} within {timeout:.1f}s. "
            f"Diagnostics: {last_diagnostics}. "
            "This usually means the CAN device is up but the Piper arm is not returning valid arm/driver feedback. "
            "Check arm power, emergency stop state, CAN cabling, and whether can_left/can_right is mapped to the correct USB-CAN adapter."
        )
    
    def set_up(self, can:str, arm_type="piper",teleop=False):
        cfg = create_agx_arm_config(robot=arm_type, channel=can, comm="can", interface="socketcan")
        self.controller = AgxArmFactory.create_arm(cfg)
        self.end_effector = self.controller.init_effector(self.controller.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.controller.connect()

        debug_print(self.name, f"robotic arm is_ok = {self.controller.is_ok()}", "INFO")
        debug_print(self.name, f"robotic gripper is_ok = {self.end_effector.is_ok()}", "INFO")

        self._wait_until_enabled()

        self.change_mode(teleop)

        debug_print(self.name, f"robotic arm is_ok = {self.controller.is_ok()}", "INFO")
        debug_print(self.name, f"robotic gripper is_ok = {self.end_effector.is_ok()}", "INFO")

    def reset(self, start_state):
        try:
            self.set_joint(start_state)
        except Exception as e:
            print(f"reset error: {e}")
        return

    # 返回单位为米
    def get_state(self):
        state = {}
        eef = self.controller.get_flange_pose().msg
        joint = self.controller.get_joint_angles().msg
        
        state["joint"] = np.array(joint)
        state["eef"] = np.array(eef)
        state["gripper"] = self.end_effector.get_gripper_status().msg.width

        return state

    # All returned values are expressed in meters,if the value represents an angle, it is returned in radians
    def set_position(self, position):
        self.controller.move_p(position)
    
    def set_joint(self, joint):
        self.controller.move_j(joint)

    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper):
        self.end_effector.move_gripper(gripper)

    def change_mode(self, teleop):
        if teleop:
            self.controller.set_leader_mode()
        else:
            self.controller.set_follower_mode()

    def __del__(self):
        try:
            if hasattr(self, 'controller'):
                # Add any necessary cleanup for the arm controller
                pass
        except:
            pass

if __name__=="__main__":
    controller = PiperController("test_piper")
    controller.set_up("can_right", teleop=False)
    controller.set_collect_info(["joint", "eef", "gripper"])
    print(controller.get())

    controller.move({
        "joint": [0., 0., 0., 0., 0., 0.],
        "gripper": 1.0,
    })
    time.sleep(2)

    controller.change_mode(teleop=True)

    while not is_enter_pressed():
        print(controller.get())
        time.sleep(1)