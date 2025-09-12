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
        self._obj_x_range = (-5, 5)
        self._obj_y_range = (-5, 5)
        self._obj_z = 0.20

    def post_reset(self):
        super().post_reset()
        self._global_step = 0  # 에피소드/런 시작 시 초기화

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_object_box(env_ids)

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        super().reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        self._randomize_object_box(env_ids)

    def _randomize_object_box(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        if n == 0:
            return
        x_min, x_max = self._obj_x_range
        y_min, y_max = self._obj_y_range

        # 1) 샘플
        x = torch.rand(n, device=self.device) * (x_max - x_min) + x_min
        y = torch.rand(n, device=self.device) * (y_max - y_min) + y_min
        z = torch.full((n,), self._obj_z, device=self.device)

        local_pos = torch.stack([x, y, z], dim=-1).contiguous()  # (n,3)
        world_pos = (self.scene.env_origins[env_ids] + local_pos).contiguous()

        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0

        # 2) 적용
        self.scene.object_box.set_world_poses(world_pos, quat, env_ids)
        self.scene.object_box.set_linear_velocities(
            torch.zeros((n, 3), device=self.device), env_ids
        )
        self.scene.object_box.set_angular_velocities(
            torch.zeros((n, 3), device=self.device), env_ids
        )

        # 3) 적용 확인(즉시 읽기)
        pos_after, quat_after = self.scene.object_box.get_world_poses()
        print(
            "[rand] first3 ids:",
            env_ids[:3].tolist(),
            " → pos:",
            pos_after[env_ids[:3]],
        )

    def step(self, actions):
        m_nav, m_align, m_grasp, _, _ = _phase_masks(self)
        m_ag = (m_align + m_grasp).clamp(0, 1).unsqueeze(-1)  # (N,1)
        ramp = 1.0 - 0.8 * m_ag  # NAV=1.0, ALIGN/GRASP=0.2

        # 스키마 인덱스
        sch = self.cfg.action_schema
        b0, bd = sch["base_cmd"]["start"], sch["base_cmd"]["dim"]

        actions = actions.clone()
        actions[:, b0 : b0 + bd] = actions[:, b0 : b0 + bd] * ramp  # 실행 감쇠
        obs, rew, terminated, truncated, info = super().step(actions)
        self._global_step += 1

        if self._global_step % 1000 == 0:
            m_nav, m_align, m_grasp, m_carry, m_place = _phase_masks(self)

            obj_pos_w, _ = _asset_root_pose_w(self, "object_box")
            ee_pos_w, _ = _body_pose_w(self, "robot", "piper_gripper_base")
            dist_ee = torch.norm(obj_pos_w - ee_pos_w, dim=-1)  # EE–Object 거리

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
                f"dist_ee_mean={dist_ee.mean():.3f}, "
                f"dist_ee_min={dist_ee.min():.3f}, "
                f"opening={opening.mean():.3f}, inside%={inside.float().mean():.2f}"
            )

        return obs, rew, terminated, truncated, info
