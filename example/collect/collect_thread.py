from robot.sensor.Cv_sensor import CvSensor
from robot.controller.Y1_controller import Y1Controller

from robot.utils.worker.worker_thread import Worker
from robot.utils.worker.time_scheduler_thread import TimeScheduler

from robot.utils.base.data_handler import is_enter_pressed
from threading import Lock
class DataBuffer:
    def __init__(self):
        self.data_buffer = {}
        self.lock = Lock()
    
    def push(self, name, data):
        with self.lock:
            if name in self.data_buffer.keys():
                self.data_buffer.append(data)
            else:
                self.data_buffer["name"] = [data]
    
    def get(self):
        return self.data_buffer
    
    def clear(self):
        self.data_buffer = {}

class component_worker(Worker):
    def __init__(self, process_name: str, start_event, end_event, component, data_buffer):
        super().__init__(process_name, start_event, end_event)
        self.component = component
        self.data_buffer = data_buffer
    
    def component_init(self):
        return
    
    def handler(self):
        data = self.get()
        self.data_buffer.push(self.component.name, data)

def main():
    arm_dict = {"left_arm": ["can1", False], "right_arm": ["can0", False]}
    arm_collect_info = ["qpos","joint", "gripper"]
    arms = {}
    for arm_name, arm_info in arm_dict.items():
        arms[arm_name] = Y1_controller[arm_name]
        arms[arm_name].set_up(*arm_info)
        arms[arm_name].set_collect_info(arm_collect_info)

    vision_dict = {
        "cam_head": [0, False, True, False],
        "cam_left_wrist": [2, False, True, False],
        "cam_right_wrist": [4, False, True, False],
    }
    vision_collect_info = ["color"]
    visions = {}

    for vision_name, vision_info in vision_dict.items():
        visions[vision_name] = CvSensor(vision_name)
        visions[vision_name].set_up(*vision_info)
        visions[vision_name].set_collect_info(vision_collect_info)

    data_buffer = DataBuffer()
    component_workers = []
    for arm in arms.values():
        


    time_scheduler = TimeScheduler()

    def reset():
        for arm in arms.values():
            move_data = {
                    "joint": [0, 0, 0, 0, 0, 0],
                    "gripper":  1.0,
                }
            arm.move(move_data)
        
        time.sleep(2)
    
    def teleop():
        for arm in arms.values():
            arm.change_mode(True)
        time.sleep(1)
    
    # 机械臂归位
    reset()

    teleop()

    
    





    