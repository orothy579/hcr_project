import gymnasium as gym
from . import agents

##
# Register Gym environments
##

gym.register(
    id="Template-Go2Piper-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)


gym.register(
    id="Template-Go2Piper-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Go2PiperBaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperBasePPORunnerCfg",
    },
)

gym.register(
    id="Template-Go2Piper-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2PiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperRoughPPORunnerCfg",
    },
)


gym.register(
    id="Template-Go2Piper-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vision_env_cfg:Go2PiperVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperVisionPPORunnerCfg",
    },
)

gym.register(
    id="Template-Go2Piper-Wholebody-v0",
    entry_point=f"{__name__}.wholebody_env:Go2PiperWholebodyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wholebody_env_cfg:Go2PiperWholebodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperWholebodyPPORunnerCfg",
    },
)
