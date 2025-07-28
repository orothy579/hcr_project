# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Project CH."""

from isaaclab.utils import configclass

# Import master configuration
try:
    from go2_piper_master.tasks.direct.go2_piper_master.go2_piper_master_env_cfg import (
        Go2PiperMasterEnvCfg,
    )
except ImportError:
    # Fallback if master project is not installed
    print("Warning: go2_piper_master not found. Using minimal fallback config.")
    from isaaclab.envs import DirectRLEnvCfg
    Go2PiperMasterEnvCfg = DirectRLEnvCfg


@configclass
class ProjectChEnvCfg(Go2PiperMasterEnvCfg):
    """Environment configuration for Project CH.
  
    Inherits from master configuration and can be customized.
    """
    action_scale: float = 1.0
    joint_gears: list = [12.0] * 12 + [1.0] * 8  # 12개의 모터는 12배 강하게, 8개의 모터는 1배 강하게
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.2
    up_weight: float = 1.0
    heading_weight: float = 1.5
    actions_cost_scale: float = 0.01
    energy_cost_scale: float = 0.02
    death_cost: float = -0.3
    alive_reward_scale: float = 0.0
    track_lin_vel_xy_exp_weight: float = 4.0
    track_ang_vel_z_exp_weight: float = 0.75
    dof_torques_l2_weight: float = -5e-5
    dof_acc_l2_weight: float = -1e-7
    feet_air_time_weight: float = 0.25
    flat_orientation_l2_weight: float = -4.0
