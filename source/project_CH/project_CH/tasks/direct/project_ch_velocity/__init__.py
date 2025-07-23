import gymnasium as gym

# CFG와 Env 클래스 import
from .config.project_ch_velocity_env_cfg import ProjectChVelocityEnvCfg
from .project_ch_velocity_env import ProjectChVelocityEnv

# 환경 등록
gym.register(
    id="Template-Project-CH-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ProjectChVelocityEnvCfg.__module__}:ProjectChVelocityEnvCfg",
    },
)
