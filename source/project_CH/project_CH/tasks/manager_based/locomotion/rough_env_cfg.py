# Auto Curriculum(비전 없음)

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)  # 기본 보상함수 있음
from project_CH.tasks.manager_based.locomotion.mdp.rewards import (
    body_relative_height,
    desired_contacts,
    undesired_contacts,
    contact_balance,
    side_support,
)

# Master 환경에서 로봇 설정 가져오기
from go2_piper_master.tasks.direct.go2_piper_master.go2_piper_master_env_cfg import (
    Go2PiperMasterEnvCfg,
)
from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from go2_piper_master.assets.go2_piper_robot import GO2_PIPER_CFG
from isaaclab.managers import ObservationTermCfg
from isaaclab.envs import mdp

# calf 와 foot 을 분리하기 위해 필요
CUSTOM_GO2_PIPER_CFG = GO2_PIPER_CFG.replace(
    spawn=GO2_PIPER_CFG.spawn.replace(merge_fixed_joints=False)
)


@configclass
class Go2PiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CUSTOM_GO2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        setattr(
            self.observations.policy,
            "zero_vision_embed",
            ObservationTermCfg(
                func="project_CH.vision.observation:zero_vision_embed",
            ),
        )

        # 초기화 시 안정적인 자세를 위해 기본 root pose와 joint pos 사용
        self.scene.robot.actuators["base_actuators"].stiffness = 20.0
        self.scene.robot.actuators["base_actuators"].damping = 3.0

        # /IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py 에 존재
        # 걷는 방향 및 목표 속도 지정
        self.commands.base_velocity.ranges.lin_vel_x = (0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

        # Height scanner 위치 지정 (base_link 에 부착)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Terrain 스케일 조정 (go2 크기에 맞춤)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains[
                "boxes"
            ].grid_height_range = (0.025, 0.1)
            self.scene.terrain.terrain_generator.sub_terrains[
                "random_rough"
            ].noise_range = (0.01, 0.06)
            self.scene.terrain.terrain_generator.sub_terrains[
                "random_rough"
            ].noise_step = 0.01

        # Action scale 조정
        self.actions.joint_pos.scale = 0.25

        # piper 관절 제외
        self.actions.joint_pos.joint_names = "FL_.*|FR_.*|HL_.*|HR_.*|piper_.*"

        # 로봇마다 거리 띄우기
        # self.scene.env_spacing = 5

        # Push 이벤트 제거
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = (
            "base_link"
        )
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0) => 초기 자세
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.weight = 0.95
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # 무릎,허벅지 닿으면 페널티
        self.rewards.undesired_contacts = RewardTermCfg(
            func=undesired_contacts,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces", body_names=".*_calf|.*_thigh"
                )
            },
        )

        # 잘 서있게 하고 싶어서... 옆으로 넘어지지마..
        self.rewards.foot_contacts = RewardTermCfg(
            func=desired_contacts,
            weight=0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces", body_names=".*_foot"
                )
            },
        )

        self.rewards.body_height = RewardTermCfg(
            func=body_relative_height,
            weight=3.0,
            params={
                "sensor_cfg": SceneEntityCfg(name="height_scanner"),
                "target_clearance": 0.42,
            },
        )

        self.rewards.contact_balance = RewardTermCfg(
            func=contact_balance,
            weight=0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces",
                )
            },
        )
        self.rewards.side_support = RewardTermCfg(
            func=side_support,
            weight=0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces",
                )
            },
        )

        # death penalty
        self.rewards.is_terminated = RewardTermCfg(
            func=mdp.is_terminated,
            weight=-30.0,
        )

        # Termination 조건 설정
        self.terminations.base_contact.params["sensor_cfg"].body_names = (
            "base_link|head_upper|head_lower|FL_hip|FR_hip|HL_hip|HR_hip|piper_base_link|piper_link1|piper_link2|piper_link3|piper_link4|piper_link5|piper_link6|piper_link7|piper_link8|piper_gripper_base"
        )
