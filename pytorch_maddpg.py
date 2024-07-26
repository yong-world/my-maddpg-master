import argparse
import os

import numpy as np
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt
import keyboard
import datetime

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_comm", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=4000, help="number of episodes")
    # parser.add_argument("--num-episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")  # 只有需要
    # 指定为DDPG算法时才需要确定这个
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="exp1", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training "
                                                                             "state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many "
                                                                    "episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and "
                                                                 "model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where "
                                                                                        "benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot "
                                                                                    "data is saved")
    parser.add_argument("--learning-curves-figure-dir", type=str, default="./learning_curves_figure/", help="learning_curves_figure_directory")
    return parser.parse_args()
def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world返回的是world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        # 构建世界，初始化随机参数，返回奖励，
        # 返回观察[自身位置 +自身速度 + 地标位置 + 其它可观智能体位置 + 其它智能体的速度 + 自身是否在森林里+通信信息]，
        # good agent的in_forest + other_vel是反过来的且没有comm
        # 返回监测(碰撞数)
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

arglist=parse_args()
env = make_env(arglist.scenario, arglist, arglist.benchmark)
obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
num_adversaries = min(env.n, arglist.num_adversaries)