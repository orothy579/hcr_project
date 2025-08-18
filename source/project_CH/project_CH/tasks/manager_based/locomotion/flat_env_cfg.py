# Baseline(고정 난이도, 비전 없음)

from isaaclab.utils import configclass
from project_CH.tasks.manager_based.locomotion.mdp.rewards import (
    undesired_contacts,
    desired_contacts,
    body_height_reward,
    suppress_leg_cross,
    feet_slide,
)
from isaaclab.managers import SceneEntityCfg, RewardTermCfg

# rough_env_cfg를 상속
from .rough_env_cfg import Go2PiperRoughEnvCfg


@configclass
class Go2PiperFlatEnvCfg(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 첫 학습 시 , 거미처럼 걷는 것을 방지 하기 위한 값 조정
        # [default] stiffness = 20 , damping = 1.0
        # self.scene.robot.actuators["base_actuators"].stiffness = 40.0
        # self.scene.robot.actuators["base_actuators"].damping = 3.0

        # /IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py 에 존재
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)

        # 보상 weight 변경 (flat 환경용)
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.track_lin_vel_xy_exp.weight = 2.5

        # Terrain을 평평한 plane으로 변경
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # Height scanner 비활성화
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # 무릎,허벅지,힙 닿으면 페널티
        self.rewards.undesired_contacts = RewardTermCfg(
            func=undesired_contacts,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces", body_names=".*_calf|.*_thigh|.*_hip"
                )
            },
        )

        # 발이 닿아있으면 보상
        self.rewards.foot_contacts = RewardTermCfg(
            func=desired_contacts,
            weight=0.25,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces", body_names=".*_foot"
                )
            },
        )

        self.rewards.body_height = RewardTermCfg(
            func=body_height_reward, weight=0.5, params={"target_height": 0.42}
        )

        self.rewards.no_cross_forward = RewardTermCfg(
            func=suppress_leg_cross, weight=-0.2, params={"vel_threshold": 0.2}
        )

        self.rewards.feet_slide = RewardTermCfg(
            func=feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces", body_names="FL_foot|FR_foot|HL_foot|HR_foot"
                ),
                "asset_cfg": SceneEntityCfg(
                    name="robot", body_names="FL_foot|FR_foot|HL_foot|HR_foot"
                ),
            },
        )


@configclass
class Go2PiperFlatEnvCfg_PLAY(Go2PiperFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Play 모드 전용 설정
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Randomization 제거
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
