# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Project CH tasks for Go2+Piper robot.
This project inherits from go2_piper_master and adds specialized tasks.
"""

import gymnasium as gym

from . import agents
from .project_ch_env import ProjectChEnv
from .project_ch_env_cfg import ProjectChEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Project-CH-Direct-v0",
    entry_point="project_CH.tasks.direct.project_ch:ProjectChEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "project_CH.tasks.direct.project_ch:ProjectChEnvCfg",
        "rsl_rl_cfg_entry_point": "project_CH.tasks.direct.project_ch.agents:rsl_rl_ppo_cfg",
    },
)