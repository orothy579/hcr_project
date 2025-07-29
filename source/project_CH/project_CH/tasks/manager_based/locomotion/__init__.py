import gym

gym.register(
    id="Go2Piper-Rough-v0",
    entry_point="manager_based.locomotion.rough_env_cfg:Go2PiperRoughEnvCfg",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "manager_based.locomotion.rough_env_cfg:Go2PiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "manager_based.locomotion.agents.rsl_rl_ppo_cfg:Go2PiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Flat-v0",
    entry_point="manager_based.locomotion.flat_env_cfg:Go2PiperFlatEnvCfg",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "manager_based.locomotion.flat_env_cfg:Go2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "manager_based.locomotion.agents.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)
