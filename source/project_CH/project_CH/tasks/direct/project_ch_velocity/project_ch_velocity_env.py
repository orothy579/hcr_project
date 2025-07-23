from isaaclab.envs import ManagerBasedRLEnv
from project_CH.tasks.direct.project_ch_velocity.config.project_ch_velocity_env_cfg import ProjectChVelocityEnvCfg

class ProjectChVelocityEnv(ManagerBasedRLEnv):
    """Velocity-based locomotion env for Project CH custom robot."""
    _ENV_CFG = ProjectChVelocityEnvCfg
