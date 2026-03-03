import torch.nn as nn
import os
import torch
import numpy as np
import pickle
import cv2
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms

from .detr.models import build_ACT_model

import IPython

e = IPython.embed

ACT_CONFIG = {
    "state_dim": 3,
    "ckpt_name": "policy_best.ckpt",
    "device": "cuda:0",
    "chunk_size": 50,
    "temporal_agg": False,
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "kl_weight": 10.0,
    "dilation": False,
    "position_embedding": "sine",
    "camera_names": ["head_cam", "left_cam", "right_cam"],
    "enc_layers": 4,
    "dec_layers": 7,
    "dim_feedforward": 3200,
    "hidden_dim": 512,
    "dropout": 0.1,
    "nheads": 8,
    "pre_norm": False,
    "masks": False,
    "pretrained_backbone": False,
}


class ACTPolicy(nn.Module):
    def __init__(self, act_args):
        super().__init__()
        self.model = build_ACT_model(act_args)  # CVAE decoder

    def __call__(self, qpos, image):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
        return a_hat


class ACT_MOBILE:
    def __init__(self, deploy_cfg, act_cfg=ACT_CONFIG):
        train_config_name = deploy_cfg.get("train_config_name")
        print(f"Use config: {train_config_name}")

        # Load statistics/checkpoint from model_path
        model_path = deploy_cfg["model_path"]
        ckpt_name = act_cfg["ckpt_name"]
        ckpt_path = os.path.join(model_path, ckpt_name)
        stats_path = os.path.join(model_path, "dataset_stats.pkl")

        self.policy = ACTPolicy(act_cfg)
        self.device = torch.device(act_cfg["device"])
        self.policy.to(self.device)
        self.policy.eval()


        # Temporal aggregation settings
        self.temporal_agg = act_cfg["temporal_agg"]
        self.num_queries = act_cfg["chunk_size"]
        self.state_dim = act_cfg["state_dim"]  # Standard joint dimension for bimanual robot
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")


        self.t = 0  # Current timestep
        self.instruction = None
        self.observation_window = None
        self.all_actions = None


        # Load dataset stats for normalization
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                self.stats = pickle.load(f)
            print(f"Loaded normalization stats from {stats_path}")
        else:
            print(f"Warning: Could not find stats file at {stats_path}")
            self.stats = None

        # Load policy weights
        if os.path.exists(ckpt_path):
            ckpt_state_dict = torch.load(ckpt_path, map_location=self.device)
            if isinstance(ckpt_state_dict, dict) and "state_dict" in ckpt_state_dict:
                ckpt_state_dict = ckpt_state_dict["state_dict"]
            loading_status = self.policy.load_state_dict(ckpt_state_dict)
            print(f"Loaded policy weights from {ckpt_path}")
            print(f"Loading status: {loading_status}")
        else:
            raise FileNotFoundError(f"Could not find policy checkpoint at {ckpt_path}")
        


    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action
    
    def infer(self, obs):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        camera_names = ["head_cam", "left_cam", "right_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)
        curr_image = curr_image / 255.0  # Normalize pixel values to [0, 1]

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        self.t += 1
        return action

    def set_language(self, instruction):
        self.instruction = instruction

    def update_obs(self, obs):
        qpos = np.concatenate([
            np.array(obs[0]["left_arm"]["joint"]).reshape(-1),
            np.array(obs[0]["left_arm"]["gripper"]).reshape(-1),
            np.array(obs[0]["right_arm"]["joint"]).reshape(-1),
            np.array(obs[0]["right_arm"]["gripper"]).reshape(-1),
            np.array(obs[0]["slamware"]["move_velocity"]).reshape(-1),
        ])

        def decode(img):
            jpeg_bytes = np.array(img).tobytes().rstrip(b"\0")
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(nparr, 1)

        img_front = decode(obs[1]["cam_head"]["color"])
        img_right = decode(obs[1]["cam_right_wrist"]["color"])
        img_left = decode(obs[1]["cam_left_wrist"]["color"])
        
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "qpos": np.asarray(qpos, dtype=np.float32),
            "head_cam": img_front.astype(np.float32),
            "left_cam": img_left.astype(np.float32),
            "right_cam": img_right.astype(np.float32),
        }
        return

    def reset(self):
        # Reset the observation cache or window here
        self.observation_window = None
        self.instruction = None
        print("successfully reset observation_window and instruction")

    
    def get_action(self, obs=None):
        if obs is not None:
            self.update_obs(obs)

        obs = self.observation_window
        
        ret_actions = []
        for _ in range(self.num_queries):
            action = self.infer(obs)
            action_vec = np.asarray(action).reshape(-1)

            if self.state_dim == 3:
                ret_actions.append({
                    "arm": {
                        "left_arm": {
                            "joint": np.zeros(6, dtype=np.float32),
                            "gripper": 0.0,
                        },
                        "right_arm": {
                            "joint": np.zeros(6, dtype=np.float32),
                            "gripper": 0.0,
                        },
                    },
                    "mobile": {
                        "slamware": {
                            "move_velocity": action_vec[:3],
                        }
                    },
                })
            elif self.state_dim == 17:
                ret_actions.append({
                    "arm": {
                        "left_arm": {
                            "joint": action_vec[:6],
                            "gripper": float(action_vec[6]),
                        },
                        "right_arm": {
                            "joint": action_vec[7:13],
                            "gripper": float(action_vec[13]),
                        },
                    },
                    "mobile": {
                        "slamware": {
                            "move_velocity": action_vec[14:17],
                        }
                    },
                })
            else:
                raise ValueError(f"unsupported action dim: {action_vec.shape[0]}")

        return ret_actions
