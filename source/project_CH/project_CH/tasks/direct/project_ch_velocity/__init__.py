import gymnasium as gym

# CFG와 Env 클래스 import
from .config.project_ch_velocity_env_cfg import ProjectChVelocityEnvCfg
from .config.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from .project_ch_velocity_env import ProjectChVelocityEnv

# 환경 등록
gym.register(
    id="Template-Project-CH-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ProjectChVelocityEnvCfg.__module__}:ProjectChVelocityEnvCfg",
        # RL library (rsl-rl) configuration entry point
        "rsl_rl_cfg_entry_point": f"{UnitreeGo2RoughPPORunnerCfg.__module__}:UnitreeGo2RoughPPORunnerCfg",
    },
)
