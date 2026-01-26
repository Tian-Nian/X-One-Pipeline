import sys
sys.path.append("./")

from threading import Event, Lock
import time
from dataclasses import dataclass, field
import math
import numpy as np

from utils.worker import Worker
# from policy.test_policy.inference_model import TestModel
from policy.pi_rtc.inference_model import PI0_DUAL
from my_robot.test_robot import TestRobot
from my_robot.y1_dual_base import Y1Dual
from utils.action_queue import ActionQueue
from utils.latency_tracker import LatencyTracker
from utils.time_scheduler import TimeScheduler
from utils.data_handler import is_enter_pressed

import torch

import json
from typing import Dict, Any

from openpi.rtc.configuration_rtc import RTCConfig, RTCAttentionSchedule

class NormStat:
    def __init__(self, *, mean=None, std=None, q01=None, q99=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std  = None if std  is None else np.asarray(std, dtype=np.float32)
        self.q01  = None if q01  is None else np.asarray(q01, dtype=np.float32)
        self.q99  = None if q99  is None else np.asarray(q99, dtype=np.float32)
    
    def _load_norm_stat(self, path):
        with open(path, "r") as f:
            raw = json.load(f)

        if "norm_stats" not in raw:
            raise KeyError("norm_stat.json must contain top-level key 'norm_stats'")

        stats_root = raw["norm_stats"]

        for key, v in stats_root.items():
            if not isinstance(v, dict):
                raise ValueError(f"Invalid norm stats for key '{key}'")

        self.mean=torch.tensor(v.get("mean"))
        self.std=torch.tensor(v.get("std"))
        self.q01=torch.tensor(v.get("q01"))
        self.q99=torch.tensor(v.get("q99"))

    def normalize_quantile(self, x):
        x = x.to('cpu')
        assert self.q01 is not None
        assert self.q99 is not None
        q01, q99 = self.q01[..., : x.shape[-1]], self.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

move_actions = []
predict_actions = []
real_actions = []
action_ptrs = []

norm_stat = NormStat()
norm_stat._load_norm_stat("/home/xspark-ai/project/openpi/checkpoint/pi05/pytorch/rtc_pi05/assets/1118/norm_stats.json")

@dataclass
class RTCDemoConfig:
    """Configuration for RTC demo with action chunking policies and real robots."""
    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            # enabled=True,
            execution_horizon=10,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    fps: float = 30.  # Action execution frequency (Hz)

    # Compute device
    device: str | None = "cuda"  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 15

def input_transform(data, size=256):
    # ====== 处理 state ======
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1),
    ])

    img_arr = [
        data[1]["cam_head"]["color"],
        data[1]["cam_right_wrist"]["color"],
        data[1]["cam_left_wrist"]["color"],
    ]

    return img_arr, state

def output_transform(data):
    if data[6]< 0.1:
        data[6] = 0.001
    
    if data[13]< 0.1:
        data[13] = 0.001
    
    move_data = {
        "arm":{
            "left_arm":{
                "joint":data[:6],
                "gripper":data[6]
            },
            "right_arm":{
                "joint":data[7:13],
                "gripper":data[13]
            }
        },
    }
    return move_data

class Number:
    def __init__(self):
        self.index = 0
    
    def clear(self):
        self.index = 0
    def play_once(self):
        self.index += 1
        # print("NOW NUMBER:", self.index)

class RobotWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, robot, action_queue: ActionQueue, number: Number):
        super().__init__(process_name, start_event, end_event)
        self.robot = robot
        self.action_queue = action_queue
        self.number = number
        self.last_move_time = time.monotonic()

    def component_init(self):
        return
    
    def handler(self):
        # 根据data_buffer获取的数据进行运动, 并
        action = self.action_queue.get()

        if action is not None:
            # now = time.monotonic()
            # if now - self.last_move_time > 5.0:
            #     print("Infering too slow, robot reset to prevent jump!")
            #     self.last_move_time = now
            #     self.robot.reset()
            #     return
            
            self.number.play_once()
            action = np.array(action.cpu())

            real_actions.append(action)

            action = output_transform(action)
            self.robot.move(action)
            print("MOVE!")
    
class InferWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, robot, policy, cfg:RTCDemoConfig, action_queue: ActionQueue,  number: Number):
        super().__init__(process_name, start_event, end_event)
        self.robot = robot
        self.action_queue = action_queue
        self.policy = policy
        self.cfg = cfg
        self.lock = Lock()
        self.number = number
        self.run = False

    def component_init(self):
        self.latency_tracker = LatencyTracker()
        self.infering = False
        self.allow_move = 0
    
    def handler(self):
        get_actions_threshold =  self.cfg.action_queue_size_to_get_new_actions
        fps = self.cfg.fps
        time_per_chunk = 1.0 / fps

        # NOT RTC
        # get_actions_threshold = 0

        print(f"TRIGGER: {self.action_queue.qsize()} / {get_actions_threshold}")
        if self.action_queue.qsize() <= get_actions_threshold: # and not self.infering:
            now = time.monotonic()
            print("start infering!")
            
            current_time = time.perf_counter()
            action_index_before_inference = self.action_queue.get_action_index()

            prev_actions = self.action_queue.get_left_over()


            if prev_actions is not None:
                # 反归一化, 使得目标约束动作与模型原本动作在同样范围
                prev_actions = norm_stat.normalize_quantile(prev_actions)
                prev_actions = prev_actions.to(self.cfg.device)
            
            inference_latency = self.latency_tracker.max()

            # print("inference_latency:", inference_latency)

            inference_delay = math.ceil(inference_latency / time_per_chunk)

            data = self.robot.get(self.run)
            # data = self.robot.get()

            if data is None:
                return
            
            img_arr, state = input_transform(data)
            self.policy.update_observation_window(img_arr, state)

            action_chunk = self.policy.get_action(inference_delay=inference_delay, 
                                                    prev_chunk_left_over=prev_actions, 
                                                    execution_horizon= self.cfg.rtc.execution_horizon)
            
            # NOT RTC
            # action_chunk = self.policy.get_action()

            if prev_actions is not None:
                move_actions.append(np.array(prev_actions.to('cpu')))
                predict_actions.append(action_chunk)
                action_ptrs.append(len(real_actions))

            original_actions = torch.tensor(action_chunk).to("cpu")
            postprocessed_actions = torch.tensor(action_chunk).to("cpu")

            # make sure inference is already arm up. 
            time_cost = time.monotonic() - now
            if time_cost < 1.0:
                self.allow_move += 1
                if self.allow_move < 5:
                    self.action_queue.reset()
                    # 回到零位做安全保护
                    action_chunk = np.tile(
                        np.array([0., 0., 0., 0., 0., 0., 1.0, 0., 0., 0., 0., 0., 0., 1.0]),
                        (50, 1)
                    )

                    original_actions = torch.tensor(action_chunk).to("cpu")
                    postprocessed_actions = torch.tensor(action_chunk).to("cpu")

                if self.allow_move == 5:
                    self.action_queue.reset()                    
                    # 回到零位做安全保护
                    # action_chunk = np.tile(
                    #     np.array([0., 0., 0., 0., 0., 0., 1.0, 0., 0., 0., 0., 0., 0., 1.0]),
                    #     (50, 1)
                    # )

                    # original_actions = torch.tensor(action_chunk).to("cpu")
                    # postprocessed_actions = torch.tensor(action_chunk).to("cpu")

                    self.run = True
                    self.action_queue.momve_gate(self.run)
                    # 重置延迟计时器
                    self.latency_tracker.reset()
                    print("START MOVE!")
            else:
                self.run = False
                self.action_queue.momve_gate(self.run)
                self.allow_move = 0

            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            self.latency_tracker.add(new_latency)

            if  self.cfg.action_queue_size_to_get_new_actions <  self.cfg.rtc.execution_horizon + new_delay:
                print(
                    "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                )
            
            self.action_queue.merge(
                original_actions, postprocessed_actions, new_delay, action_index_before_inference
            )
            print(f"Infer success! Time cost:{time_cost}")
            print("new_latency:", new_latency)
            self.number.clear()

import numpy as np
import matplotlib.pyplot as plt

def plot_action_alignment(
    real_actions: list[np.ndarray],
    predict_actions: list[np.ndarray],
    move_actions: list[np.ndarray],
    action_ptrs: list[int],
    save_path: str = "action_alignment.png",
):
    real_arr = np.stack(real_actions)  # (T_real, D)
    T_real, D = real_arr.shape

    fig, axes = plt.subplots(D, 1, figsize=(14, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    time_real = np.arange(T_real)

    for d in range(D):
        ax = axes[d]

        # ---- real trajectory ----
        ax.plot(
            time_real,
            real_arr[:, d],
            color="black",
            linewidth=1.5,
            label="real",
        )

        for k, (pred, move, ptr) in enumerate(
            zip(predict_actions, move_actions, action_ptrs)
        ):
            if ptr >= T_real:
                continue

            # ===== predict =====
            if pred is not None and len(pred) > 0:
                Lp = min(pred.shape[0], T_real - ptr)
                if Lp > 0:
                    t_pred = ptr + np.arange(Lp)
                    ax.plot(
                        t_pred,
                        pred[:Lp, d],
                        color="tab:blue",
                        linewidth=1.2,
                        alpha=0.7,
                        label="predict" if k == 0 else None,
                    )

            # ===== move =====
            if move is not None and len(move) > 0:
                Lm = min(move.shape[0], T_real - ptr)
                if Lm > 0:
                    t_move = ptr + np.arange(Lm)
                    ax.plot(
                        t_move,
                        move[:Lm, d],
                        color="tab:orange",
                        linewidth=1.2,
                        alpha=0.7,
                        label="move" if k == 0 else None,
                    )

            # ---- inference trigger ----
            ax.axvline(
                ptr,
                color="red",
                linewidth=0.8,
                alpha=0.25,
            )

        ax.set_ylabel(f"dim {d}")
        ax.grid(True, alpha=0.2)

        if d == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[align] plot saved to: {save_path}")

def main_cli():
    cfg = RTCDemoConfig()

    start_event, end_event = Event(), Event()
    robot = Y1Dual()# TestRobot()
    # robot = TestRobot(replay_path="save/deploy/16.hdf5")
    # robot = TestRobot(replay_path="save/complete_traj_1117/20.hdf5")
    robot.set_up()
    robot.reset()

    action_queue = ActionQueue(cfg.rtc)

    number = Number()
    robot_worker = RobotWorker("robot", start_event, end_event, robot, action_queue, number)

    policy = PI0_DUAL("/home/xspark-ai/project/control_your_robot/policy/openpi/checkpoint/pi05/pytorch/rtc_pi05/", "test", rtc=True)
    policy.set_instruction("Open the box, take out the bag inside the box, place it on the right side, fold the box.")
    # policy = PI0_DUAL("/home/xspark-ai/project/openpi/checkpoint/airpods/", "dec18", rtc=False)
    # policy = None
    # warm_up(policy, 1.5)
    infer_worker = InferWorker("infer", start_event, end_event, robot, policy, cfg, action_queue, number)

    time_scheduler_robot = TimeScheduler(work_events=[robot_worker.forward_event], time_freq=30, end_events=[robot_worker.next_event], process_name="time_scheduler_robot")
    time_scheduler_infer = TimeScheduler(work_events=[infer_worker.forward_event], time_freq=30, end_events=[infer_worker.next_event], process_name="time_scheduler_infer")

    robot_worker.start()
    infer_worker.start()

    is_start = False

    while not is_start:
        time.sleep(0.01)
        if is_enter_pressed():
            is_start = True
            start_event.set()
        else:
            time.sleep(1)

    time_scheduler_robot.start()
    time_scheduler_infer.start()
    
    while is_start:
        time.sleep(0.01)
        if is_enter_pressed():            
            end_event.set()  
            time.sleep(2)
            # plot_prev_and_predict(predict_actions=predict_actions, prev_actions=move_actions)
            plot_action_alignment(real_actions=real_actions, predict_actions=predict_actions, move_actions=move_actions, action_ptrs=action_ptrs)
            robot.draw("4.png")

            time_scheduler_robot.stop()  
            time_scheduler_infer.stop()  
            is_start = False

    # 给数据写入一定时间缓冲
    time.sleep(1)

    robot_worker.stop()
    infer_worker.stop()

if __name__ == "__main__":
    main_cli()