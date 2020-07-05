import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np
from argparse import Namespace

import dmc2gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder

from utils_fork import parse_opts, flatten_cfg

torch.backends.cudnn.benchmark = True


extra_cfg = None


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])


    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def adjust_action_repeat_hack(cfg):
    if cfg.env == 'walker_walk':
        cfg.action_repeat = 2
    elif cfg.env == 'reacher_easy':
        cfg.action_repeat = 4
    elif cfg.env == 'finger_spin':
        cfg.action_repeat = 2
    elif cfg.env == 'cheetah_run':
        cfg.action_repeat = 4
    elif cfg.env == 'ball_in_cup_catch':
        cfg.action_repeat = 4
    elif cfg.env == 'cartpole_swingup':
        cfg.action_repeat = 8
    else:
        raise NotImplementedError


class Workspace(object):
    def __init__(self, cfg):

        self.work_dir = os.getcwd()

        """Hack to adjust action_repeat"""
        adjust_action_repeat_hack(cfg)

        print(f"CFG:\n{'-'*100}\n{cfg}\n{'-'*100}")

        self.cfg = cfg
        experiment_name = f"{cfg.full_title}_{cfg.run_id}"

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             save_wb=cfg.log_save_wandb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat,
                             cfg=dict(flatten_cfg(cfg)),
                             plot_project="drqtest",
                             experiment=experiment_name)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        print(f"ACTOR:\n{'-'*100}\n{self.agent.actor}\n{'-'*100}")
        print(f"CRITIC:\n{'-'*100}\n{self.agent.critic}\n{'-'*100}")

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device,
                                          use_aug=cfg.replay_buffer_augmentation)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


# @hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    # -- Pre-process cfg
    if cfg.seed == 0:
        cfg.seed = cfg.run_id + 1

    workspace = Workspace(cfg)
    workspace.run()


def run(cfg):
    # Hack to load with liftoff
    import sys

    # Hack to move configs from liftoff
    sys.argv = [sys.argv[0], ] + [f"{k}={v}" for k, v in flatten_cfg(cfg)]

    main_wrapper = hydra.main('config.yaml', strict=False)
    main_wrapper(main)()


if __name__ == "__main__":
    import sys

    cfg = parse_opts()

    # Hack to move configs from liftoff
    ftcfg = flatten_cfg(cfg)
    # ftcfg = [('seed', 0), ('log_save_tb', False), ('save_video', False), ('log_save_wandb', False), ('replay_buffer_augmentation', True), ('out_dir', 'results/experiment_configs'), ('run_id', 1)]
    sys.argv = [sys.argv[0], ] + [f"{k}={v}" for k, v in ftcfg]

    main_wrapper = hydra.main('config.yaml', strict=False)
    main_wrapper(main)()
