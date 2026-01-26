import sys
sys.path.append('./')

import os
import importlib
import argparse
import numpy as np
import time
import yaml
import json
import torch

from typing import Dict, Optional, Tuple
from src.robot.utils.base.data_handler import debug_print, is_enter_pressed

import sys
sys.path.insert(0, "/home/xspark-ai/project/control_your_robot/src/robot/policy/MemoryMatters_VLA")

from scripts.image_utils import to_pil
from scripts.layout_utils import env_to_model_layout, model_to_env_layout
from scripts.normlization import denormalize_arms, load_stats, normalize_arms
from source.agent import MemoryMattersAgent

from termcolor import cprint

# START ================ you could modify to your format ================ 

fps = 30
import cv2

DIM_FLAG = 0
# Enable quantile-based normalization by default; set to False to use mean/std.
QUANTILE = False

# Runtime knobs filled by get_model, read inside encode_obs
_RUNTIME_SETTINGS: Dict[str, object] = {
    "camera_key": "head_camera",
    "image_size": (224, 224),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Stats for normalization/denormalization (arms only; grippers already 0~1 from env)
_STATS: Dict[str, Optional[np.ndarray]] = {
    "state_mean": None,
    "state_std": None,
    "action_mean": None,
    "action_std": None,
    "state_min": None,
    "state_max": None,
    "action_min": None,
    "action_max": None,
    "state_q01": None,
    "state_q99": None,
    "action_q01": None,
    "action_q99": None,
}

def _extract_state(observation: dict) -> np.ndarray:
    """Extract 16-d joint vector; pad or truncate to 16 dims."""
    joint_vec = observation.get("joint_action", {}).get("vector")
    if joint_vec is None:
        cprint("[deploy] joint_action.vector missing, using zeros", "yellow")
        return np.zeros((16,), dtype=np.float32)
    joint_arr = np.asarray(joint_vec, dtype=np.float32).reshape(-1)
    
    if joint_arr.size < 16:
        padded = np.zeros((16,), dtype=np.float32)
        padded[ : joint_arr.size // 2 - 1] = joint_arr[ : joint_arr.size // 2 - 1]
        padded[7] = joint_arr[joint_arr.size // 2 - 1] 
        padded[8 : 8 + joint_arr.size // 2 - 1] = joint_arr[joint_arr.size // 2 : -1]
        padded[15] = joint_arr[joint_arr.size - 1] 
        
        joint_arr = padded
        
        global DIM_FLAG
        DIM_FLAG = 1
        
        # cprint(f"[deploy] joint vector padded to 16 dims (got {joint_arr.size})", "yellow")
        
    elif joint_arr.size > 16:
        joint_arr = joint_arr[:16]
        cprint("[deploy] joint vector truncated to 16 dims", "yellow")
        
    return joint_arr


def _normalize_state(state_vec: np.ndarray) -> np.ndarray:
    """Normalize state (model layout) using quantile or mean/std."""
    if QUANTILE:
        q01 = _STATS.get("state_q01")
        q99 = _STATS.get("state_q99")
        if q01 is not None and q99 is not None:
            return normalize_arms(
                state_vec,
                None,
                None,
                arm_dims=state_vec.shape[-1],
                quantile=True,
                q01=q01,
                q99=q99,
            )
    
    # print (f"_STATS = {_STATS}")
    min = _STATS.get("state_min")
    max = _STATS.get("state_max")
    if min is not None and max is not None:
        return normalize_arms(state_vec, None, None, min, max, arm_dims=14)

    mean = _STATS.get("state_mean")
    std = _STATS.get("state_std")
    if mean is None or std is None:
        return state_vec
    return normalize_arms(state_vec, mean, std, arm_dims=14)


def _denormalize_action(action_vec: np.ndarray) -> np.ndarray:
    """Denormalize actions using quantile or mean/std; grippers stay untouched after denorm."""
    if QUANTILE:
        q01 = _STATS.get("action_q01")
        q99 = _STATS.get("action_q99")
        if q01 is not None and q99 is not None:
            return denormalize_arms(
                action_vec,
                None,
                None,
                arm_dims=action_vec.shape[-1],
                quantile=True,
                q01=q01,
                q99=q99,
            )
    
    min = _STATS.get("action_min")
    max = _STATS.get("action_max")
    if min is not None and max is not None:
        return denormalize_arms(action_vec, None, None, min, max, arm_dims=14)

    mean = _STATS.get("action_mean")
    std = _STATS.get("action_std")
    if mean is None or std is None:
        return action_vec
    return denormalize_arms(action_vec, mean, std, arm_dims=14)


def _load_stats(stats_path: str) -> None:
    """
    Load state/action stats from JSON.
    Supports mean/std or quantile stats (q01/q99).
    """
    if not stats_path:
        cprint("[deploy] stats path not provided; skipping stats load", "yellow")
        return
    try:
        stats = load_stats(stats_path)
        _STATS.update(stats)
        has_quantile = (
            _STATS["state_q01"] is not None
            and _STATS["state_q99"] is not None
            and _STATS["action_q01"] is not None
            and _STATS["action_q99"] is not None
        )
        has_mean_std = _STATS["state_mean"] is not None and _STATS["action_mean"] is not None
        has_min_max = _STATS["state_min"] is not None and _STATS["action_min"] is not None

        cprint(f"[deploy] loaded stats from {stats_path}", "cyan")
        if has_quantile:
            cprint(
                f"[deploy] quantile state q01 head={_STATS['state_q01'][:3]}; q99 head={_STATS['state_q99'][:3]}",
                "cyan",
            )
        if has_mean_std:
            cprint(
                f"[deploy] state_std head={_STATS['state_std'][:3] if _STATS['state_std'] is not None else 'None'}",
                "cyan",
            )
        if has_min_max:
            cprint(
                f"[deploy] min/max state min head={_STATS['state_min'][:3]}; max head={_STATS['state_max'][:3]}",
                "cyan",
            )
    except Exception as exc:
        cprint(f"[deploy] failed to load stats ({stats_path}): {exc}", "red")


def _postprocess_action_chunk(actions_model: np.ndarray) -> np.ndarray:
    """
    Denormalize arms, reorder layout, clip grippers, and return env-ready chunk.
    actions_model: (T, 16) in model layout (normalized), optionally with leading batch/horizon dims.
    """
    chunk = np.array(actions_model, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    flat_actions = chunk.reshape(-1, chunk.shape[-1])

    np.set_printoptions (precision = 4, suppress = True)
    # print (f"ori_chunk = {chunk}")
    # print (f"DIM_FLAG = {DIM_FLAG}")

    processed = []
    for step in flat_actions:
        denorm = _denormalize_action(step)
        env_order = model_to_env_layout(denorm)
        # env_order[7] = np.clip(env_order[7], 0.0, 1.0)
        # env_order[15] = np.clip(env_order[15], 0.0, 1.0)
        # print (f"ori: {env_order}")
        if DIM_FLAG == 1:
            env_order = np.concatenate((env_order[0 : 6], env_order[7 : 8], env_order[8 : 14], env_order[15 : 16]), axis = 0)
        # print (f"final: {env_order}")
        processed.append(env_order)
    processed = np.stack(processed, axis=0)
    # if processed.size:
    #     cprint(f"[deploy] action postprocess: env arm[0] sample = {processed[0][:4]}", "blue")
    return processed

def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode('.jpg', imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b'\0'))
    return encode_data, max_len

def encode_obs(observation: dict) -> Dict[str, object]:
    """
    Post-process raw observation into model-ready payload:
    - pick camera
    - reorder state to model layout
    - normalize arm dims with stats
    """
    cam_key: str = _RUNTIME_SETTINGS.get("camera_key", "head_camera")
    target_size: Tuple[int, int] = _RUNTIME_SETTINGS.get("image_size", (224, 224))

    # Camera selection with fallback
    image_array = None
    # image_array = observation["third_view_rgb"] if "third_view_rgb" in observation else None
    obs_block = observation.get("observation", {})
    if cam_key in obs_block and isinstance(obs_block[cam_key], dict) and "rgb" in obs_block[cam_key]:
        image_array = obs_block[cam_key]["rgb"]
    else:
        # pick the only/first camera that has rgb
        rgb_candidates = [(name, payload) for name, payload in obs_block.items() if isinstance(payload, dict) and "rgb" in payload]
        if rgb_candidates:
            name, payload = rgb_candidates[0]
            image_array = payload["rgb"]
        else:
            raise KeyError("No RGB camera found in observation")
    if image_array is None:
        raise KeyError("No RGB camera found in observation payload")

    pil_image = to_pil(np.array(image_array), target_size)
    raw_state_env = _extract_state(observation)
    model_state = env_to_model_layout(raw_state_env)

    # np.set_printoptions (precision = 5, suppress = True)
    # print (f"raw = {model_state}")

    norm_stats = _normalize_state(model_state)
    # print (f"norm_stats = {norm_stats}")
    # cprint(f"[deploy] obs: cam = {cam_key}, state (norm) head = {norm_stats[:4]}", "cyan")
    return {"image": pil_image, "state": norm_stats.reshape(1, -1), "instruction": observation.get("instruction", "")}


def input_transform(data):
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
    ])

    # ====== 处理图像 ======
    img_arr = [
        data[1]["cam_head"]["color"],
        data[1]["cam_right_wrist"]["color"],
        data[1]["cam_left_wrist"]["color"],
    ]

    # img_enc, img_enc_len = images_encoding(img_arr)
    return img_arr, state
    # return img_enc, state

def output_transform(data):
    if data[6] < 0.5:
        data[6] -= 0.05
    
    if data[13] < 0.5:
        data[13] -= 0.05

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
        }
    }
    return move_data

# END ================ you could modify to your format ================ 
def get_class(import_name, class_name):
    try:
        class_module = importlib.import_module(import_name)
        debug_print("function", f"Module loaded: {class_module}", "DEBUG")
    except ModuleNotFoundError as e:
        raise SystemExit(f"ModuleNotFoundError: {e}")

    try:
        return_class = getattr(class_module, class_name)
        debug_print("function", f"Class found: {return_class}", "DEBUG")

    except AttributeError as e:
        raise SystemExit(f"AttributeError: {e}")
    except Exception as e:
        raise SystemExit(f"Unexpected error instantiating model: {e}")
    return return_class


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the task")
    parser.add_argument("--base_model_class", type=str, required=True, help="Name of the model class")
    parser.add_argument("--base_model_path", type=str, required=True, help="model path, e.g., policy/RDT/checkpoints/checkpoint-10000. If using RoboTwin pipeline, this should be set as checkpoint_id")
    parser.add_argument("--base_task_name", type=str, required=True, help="task name, read intructions from task_instuctions/{base_task_name}.json")
    parser.add_argument("--base_robot_name", type=str, required=True, help="robot name, read my_robot/{base_robot_name}.py")
    parser.add_argument("--base_robot_class", type=str, required=True, help="robot class, get class from my_robot/{base_robot_name}.py")
    parser.add_argument("--episode_num", type=int, default=10000, help="how many episodes you want to deploy")
    parser.add_argument("--max_step", type=int, default=1000000000000, help="the maximum step for each episode")
    parser.add_argument("--robotwin", action="store_true", help="If using RoboTwin pipeline, you should set it.")
    parser.add_argument("--video", type=str, default=None, help="Recording the video if set, should set to cam_name like cam_head.")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    args_dict = vars(args)

    # ---------- 读取 YAML 配置 ----------
    def load_yaml_safe(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        return {}

    # 分别读取两个配置文件
    robotwin_path = "config/RoboTwin_setting.yml"
    base_model_path = f"src/robot/policy/{args.base_model_name}/deploy_policy.yml"

    robotwin_setting = load_yaml_safe(robotwin_path)
    model_setting = load_yaml_safe(base_model_path)

    # ---------- 合并配置 ----------
    # 优先级顺序：
    # 命令行参数 < robotwin_setting < model_setting < overrides
    merged = {}
    merged.update(args_dict)
    merged.update(robotwin_setting)
    merged.update(model_setting)

    # ---------- 解析 overrides ----------
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        merged.update(overrides)

    # 返回合并后的结果（dict）
    return merged

# ROboTwin eval
class TASK_ENV:
    def __init__(self, base_task_name):
        self.base_task_name = base_task_name
    
    def get_instruction(self):
        json_Path = os.path.join( "task_instructions", f"{self.base_task_name}.json")
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict['instructions']
        instruction = np.random.choice(instructions)
        return instruction

class RoboTwinModel:
    def __init__(self, model, encode_obs, base_task_name):
        self.model = model
        self.encode_obs = encode_obs
        self.TASK_ENV = TASK_ENV(base_task_name)
        self.instruction = ""
    
    def random_set_language(self):
        self.instruction = self.TASK_ENV.get_instruction()
        debug_print("RoboTwinModel", "Eval under RoboTwin pipeline, set instruction by policy/{model}/deploy_policy.py", "DEBUG")
        return
    
    def update_observation_window(self, img_arr, state):
        self.observation_window = {}

        self.observation_window["observation"] = {}
        self.observation_window["observation"]["head_camera"] = {"rgb": img_arr[0]}
        self.observation_window["observation"]["right_camera"] = {"rgb": img_arr[1]}
        self.observation_window["observation"]["left_camera"] = {"rgb": img_arr[2]}
        self.observation_window["agent_pos"] = state
        self.observation_window["joint_action"] = {"left_arm": state[:6],
                                                   "left_gripper": state[6],
                                                   "right_arm": state[7:13],
                                                   "right_gripper": state[13],
                                                   "vector": state,}

    def update_obs(self, obs):
        self.model.update_obs(obs)

    def get_action(self):
        # obs = self.encode_obs(self.observation_window)
        # # ======== Get Action ========
        # self.model.update_obs(obs)
        actions = self.model.get_action()
        # import pdb;pdb.set_trace()
        return actions
    
    def reset_obsrvationwindows(self):
        # self.model.reset_obsrvationwindows()
        self.model.reset()
        return
    
def init():
    args = parse_args_and_config()

    is_robotwin = args["robotwin"]
    is_video = args["video"]
    
    if not is_robotwin:
        base_model_class = get_class(f"robot.policy.{args['base_model_name']}.inference_model", args["base_model_class"])
        model = base_model_class(args["base_model_path"], args["base_task_name"], **args)
    else:
        get_model = get_class(f"robot.policy.{args['base_model_name']}.deploy_policy", "get_model")
        encode_obs = get_class(f"robot.policy.{args['base_model_name']}.deploy_policy", "encode_obs")
        base_model = get_model(args)
        stats_path = args.get("state_stats_path", "")
        _load_stats(stats_path)
        model = RoboTwinModel(base_model, encode_obs, args["base_task_name"])
        
    base_robot_class = get_class(f"my_robot.{args['base_robot_name']}", args["base_robot_class"])
    robot = base_robot_class()

    return model, robot, args["episode_num"], args["max_step"], is_video, args

if __name__ == "__main__":
    os.environ["INFO_LEVEL"] = "INFO" # DEBUG , INFO, ERROR
    
    model, robot, episode_num, max_step, video_cam_name, args = init()
    robot.set_up()
    video_path=f"save/videos/{args['base_task_name']}"
    for i in range(episode_num):
        step = 0
        # 重置所有信息
        robot.reset()
        model.reset_obsrvationwindows()
        model.random_set_language()

        writer = None
        if video_cam_name is not None:
            import cv2
            first_frame = robot.get()[1][video_cam_name]["color"][:,:,::-1]
            height, width, channels = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 'XVID'
            video_dir = video_path + f"/{i}/"
            os.makedirs(video_dir, exist_ok=True)
            writer = cv2.VideoWriter(os.path.join(video_dir, f"{video_cam_name}.mp4"), fourcc, fps, (width, height))
            print(f"Video saving enabled: {video_path}, fps={fps}, size=({width},{height})")

        # 等待允许执行推理指令, 按enter开始
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("start to inference, press ENTER to end...")
            else:
                print("waiting for start command, press ENTER to star...")
                time.sleep(1)

        # 开始逐条推理运行
        while step < max_step and is_start:
            data = robot.get()
            img_arr, state = input_transform(data)
            model.update_observation_window(img_arr, state)
            
            observation = model.observation_window

            if model.model.is_init == 0:
                model.model.is_init = 1
                image = observation["observation"]["head_camera"]["rgb"]
                model.model.init_high_with_image()
                
                instruction = model.TASK_ENV.get_instruction()
                
            instruction = model.TASK_ENV.get_instruction()
            observation["instruction"] = instruction
            encoded_obs = encode_obs(observation)

            if model.model.action_count == 0:
                model.update_obs(encoded_obs)

            result = model.get_action()

            if result is None or len(result) == 0:
                cprint("[deploy] no actions produced", "red")
                raise SystemExit("Empty actions from model; aborting eval.")
            else:
                # Accumulate current normalized prediction for absolute times [iter, iter + action_horizon)
                model.model.accumulate_actions_chunk(result["normalized_actions"])  # updates internal time-indexed history
                # Build smoothed normalized actions for [iter, iter + action_strip)
                smoothed_model_actions = model.model.get_smoothed_actions(model.model.iter, model.model.action_strip)
                # Postprocess smoothed actions (denormalize + layout reorder)
                actions = _postprocess_action_chunk(smoothed_model_actions)
                # cprint(f"[deploy] action chunk env layout shape = {actions.shape}", "cyan")
            
            # import time
            # time.sleep (1000)
            steps_to_run = min(model.model.action_strip, actions.shape[0])

            for idx in range(steps_to_run):
                action = actions[idx]
                if video_cam_name is not None:
                    frame = robot.get()[1][video_cam_name]["color"]
                    writer.write(frame)
                
                if step % 10 == 0:
                    debug_print("main", f"step: {step}/{max_step}", "INFO")
                move_data = output_transform(action)
                robot.move(move_data)
                step += 1
                # time.sleep(1/robot.condition["save_freq"])
                time.sleep(1 / 25)
                # if st
                # 
                # ep >= max_step or is_enter_pressed():
                
                # ============
                data = robot.get()
                img_arr, state = input_transform(data)
                model.update_observation_window(img_arr, state)
                
                observation = model.observation_window

                if model.model.is_init == 0:
                    model.model.is_init = 1
                    image = observation["observation"]["head_camera"]["rgb"]
                    model.model.init_high_with_image()
                    
                    instruction = model.TASK_ENV.get_instruction()
                    
                instruction = model.TASK_ENV.get_instruction()
                observation["instruction"] = instruction
                encoded_obs = encode_obs(observation)

                model.update_obs(encoded_obs)
                # ============

                if is_enter_pressed():
                    debug_print("main", "enter pressed, the episode end", "INFO")
                    is_start = False
                    break
            
            model.model.iter += steps_to_run

        if writer is not None:
            writer.release()
        debug_print("main",f"finish episode {i}, running steps {step}","INFO")