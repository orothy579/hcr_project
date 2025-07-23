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
        "rsl_rl_cfg_entry_point": "project_CH.tasks.direct.project_ch.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

import importlib, pkg_resources, sys, inspect, pprint

print("=== pip list 에서 확인 ===")
print([p.project_name for p in pkg_resources.working_set if "go2-piper" in p.project_name.lower()])

print("\n=== import 테스트 ===")
try:
    m = importlib.import_module("go2_piper_master")
    print("모듈 위치 :", m.__file__)
    print("하위 속성  :", dir(m)[:10], "...")
except ImportError as e:
    print("ImportError:", e)

print("\n=== sys.path ===")
pprint.pprint(sys.path[:5])