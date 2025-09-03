from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class Go2PiperWholeBodyEnv(ManagerBasedRLEnv):
    def step(self, actions):
        # === 내가 추가한 훅 ===
        if getattr(self.cfg, "stage_mode", "") == "pretrain":
            # 팔/그리퍼 액션 0으로 고정
            sch = self.cfg.action_schema
            a0, d0 = sch["arm_delta"]["start"], sch["arm_delta"]["dim"]
            g0, gd = sch["gripper"]["start"], sch["gripper"]["dim"]
            actions = actions.clone()
            actions[:, a0 : a0 + d0] = 0.0
            actions[:, g0 : g0 + gd] = 0.0

        # === 원래 ManagerBasedRLEnv의 step() 실행 ===
        return super().step(actions)
