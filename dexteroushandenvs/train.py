# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL

import torch


def train():
    print("Algorithm: ", args.algo)
    # Agent: 4x3
    # agent_index = [[[0, 1, 2],[ 3, 4, 5]],
    #                [[0, 1, 2],[ 3, 4, 5]]]
    # Agent: 2x6
    agent_index = [[[0, 1, 2, 3, 4, 5]],
                   [[0, 1, 2, 3, 4, 5]]]

    if args.algo in ["mappo", "happo","ippo", "macpo", "mappolag"]: 
        # maddpg exists a bug now 
        args.task_type = "MultiAgent"

        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        runner = process_MultiAgentRL(args,env=env, config=cfg_train, model_dir=args.model_dir)
        
        if args.model_dir != "":
            runner.eval(100000)
        else:
            runner.run()

    

    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [mappo, happo, ippo, macpo, mappolag]")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
