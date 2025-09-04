import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

# rewards.py 유틸
from project_CH.tasks.manager_based.locomotion.mdp.rewards import (
    _phase_masks,
    _asset_root_pose_w,
    _body_pose_w,
    get_gripper_opening_generic,
)


class Go2PiperWholebodyEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._global_step = 0  # 전역 스텝 카운터

    def post_reset(self):
        super().post_reset()
        self._global_step = 0  # 에피소드/런 시작 시 초기화

    def step(self, actions):
        # 표준 5튜플
        obs, rew, terminated, truncated, info = super().step(actions)
        self._global_step += 1

        # 1000 스텝마다 게이팅/상태 로그
        if self._global_step % 1000 == 0:
            m_nav, m_align, m_grasp, m_carry, m_place = _phase_masks(self)

            obj_pos_w, _ = _asset_root_pose_w(self, "object_box")
            ee_pos_w, _ = _body_pose_w(self, "robot", "piper_gripper_base")
            dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)

            opening = get_gripper_opening_generic(self).squeeze(-1)

            zone_pos_w, _ = _asset_root_pose_w(self, "place_zone")
            inside = ((obj_pos_w[:, 0] - zone_pos_w[:, 0]).abs() <= 0.25) & (
                (obj_pos_w[:, 1] - zone_pos_w[:, 1]).abs() <= 0.25
            )

            print(
                f"[step {self._global_step}] "
                f"m(nav,align,grasp,carry,place)="
                f"{m_nav.mean():.2f}, {m_align.mean():.2f}, {m_grasp.mean():.2f}, "
                f"{m_carry.mean():.2f}, {m_place.mean():.2f} | "
                f"dist={dist.mean():.3f}, opening={opening.mean():.3f}, inside%={inside.float().mean():.2f}"
            )

        return obs, rew, terminated, truncated, info
