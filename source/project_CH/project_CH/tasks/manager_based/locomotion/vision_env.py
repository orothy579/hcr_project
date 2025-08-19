import math
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from torch import nn

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat  # (roll, pitch, yaw)
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg


from project_CH.tasks.manager_based.locomotion.vision_env_cfg import (
    Go2PiperVisionEnvCfg,
)
from project_CH.vision import TerrainCNN


def preprocess_rgb(env, cam_name: str, out_size: int = 96) -> torch.Tensor:
    """
    Isaac Lab 권장 API로 카메라 이미지를 가져와 (B,3,H,W) float32 [0,1] 형태로 리턴.
    - cam_name: SceneCfg에 선언한 센서 속성명 (예: "ee_cam")
    - out_size: 정사각 리사이즈 크기
    """
    # (B,H,W,3), 기본 normalize=True → [0,1] 스케일 + 배치 평균 제거
    img_bhwc = mdp.observations.image(
        env,
        sensor_cfg=SceneEntityCfg(name=cam_name),
        data_type="rgb",
        normalize=True,
    ).to(
        env.device
    )  # (B,H,W,3), float32
    # (B,3,H,W)로 변환
    img_bchw = img_bhwc.permute(0, 3, 1, 2).contiguous()
    # 크기 맞추기
    if out_size is not None and (
        img_bchw.shape[-2] != out_size or img_bchw.shape[-1] != out_size
    ):
        img_bchw = F.interpolate(
            img_bchw, size=(out_size, out_size), mode="bilinear", align_corners=False
        )
    return img_bchw


class EpisodeMeter:
    """
    에피소드 메트릭 수집기 (5지표를 episode 단위로 누적/집계)
    수집 지표:
      - Success Rate
      - V_MAE (cmd vs body xy)
      - Tilt RMS (roll,pitch)
      - Slip Ratio (접지 중 |v_xy| > τ 인 step 비율)
      - Forward Progress (x_now - x0)
    """

    def __init__(self, num_envs, device, cfg=None):
        self.device = device
        self.num_envs = num_envs
        self.cfg = SimpleNamespace(
            success_v_mae_max=0.25,  # m/s
            success_prog_min=2.0,  # m
            slip_speed_thresh=0.2,  # m/s
            **(cfg.__dict__ if cfg is not None else {})
        )
        self.reset_buffers()

    def reset_buffers(self, env_ids=None):
        if env_ids is None:
            n = self.num_envs
            self.cnt = torch.zeros(n, device=self.device)
            self.sum_stability = torch.zeros(n, device=self.device)  # 참고용(로그)
            self.sum_v_mae = torch.zeros(n, device=self.device)
            self.sum_tilt_sq = torch.zeros(n, device=self.device)
            self.slip_cnt = torch.zeros(n, device=self.device)
            self.contact_steps = torch.zeros(n, device=self.device)
            self.x0 = torch.full((n,), torch.nan, device=self.device)
            self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            self.cnt[env_ids] = 0
            self.sum_stability[env_ids] = 0
            self.sum_v_mae[env_ids] = 0
            self.sum_tilt_sq[env_ids] = 0
            self.slip_cnt[env_ids] = 0
            self.contact_steps[env_ids] = 0
            self.x0[env_ids] = torch.nan
            self.success[env_ids] = False

    @torch.no_grad()
    def step_accumulate(self, env, env_ids=None):
        robot = env.scene["robot"]
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.cnt[env_ids] += 1

        # 초기 x 포지션 저장
        root_pos = robot.data.root_pos_w[:, :3]
        new_mask = torch.isnan(self.x0[env_ids])
        if new_mask.any():
            self.x0[env_ids[new_mask]] = root_pos[env_ids[new_mask], 0]

        # Stability(평균만 참고 로그) — env.metrics에 계산돼 있다고 가정
        if hasattr(env, "metrics") and ("stability_score" in env.metrics):
            self.sum_stability[env_ids] += env.metrics["stability_score"][
                env_ids
            ].float()

        # Velocity MAE (xy)
        cmd = env.command_manager.get_command("base_velocity")  # (N,3): [vx, vy, wz]
        v_cmd_xy = cmd[env_ids, :2]
        v_body_xy = robot.data.root_lin_vel_w[env_ids, :2]
        v_mae = (v_cmd_xy - v_body_xy).abs().mean(dim=-1)
        self.sum_v_mae[env_ids] += v_mae

        # Tilt (roll,pitch) RMS
        q = robot.data.root_quat_w[env_ids, :]
        rpy = euler_xyz_from_quat(q)  # (M,3)
        tilt_sq = rpy[:, 0] ** 2 + rpy[:, 1] ** 2
        self.sum_tilt_sq[env_ids] += tilt_sq

        # Slip ratio: 접지 중 |v_xy| > τ 이면 slip
        foot_names = ["FL_foot", "FR_foot", "HL_foot", "HR_foot"]
        names = robot.data.body_names
        foot_ids = [names.index(n) for n in foot_names if n in names]

        if len(foot_ids) > 0 and "contact_forces" in env.scene.sensors:
            v_xy = robot.data.body_lin_vel_w[env_ids][:, foot_ids, :2].norm(
                dim=-1
            )  # (M,4)

            cs: ContactSensor = env.scene.sensors["contact_forces"]
            try:
                netF = cs.data.net_forces_w[:, foot_ids, :]  # (N,4,3)
                contact = netF[env_ids].norm(dim=-1) > 1.0  # (M,4)
            except Exception:
                netF = cs.data.net_forces_w_history[:, -1, foot_ids, :]  # (N,4,3)
                contact = netF[env_ids].norm(dim=-1) > 1.0

            contacting = contact.any(dim=1)  # (M,)
            self.contact_steps[env_ids] += contacting.float()

            slip_now = ((v_xy > self.cfg.slip_speed_thresh) & contact).any(dim=1)
            self.slip_cnt[env_ids] += slip_now.float()

    @torch.no_grad()
    def finalize_episode(self, env, env_ids, early_terminated_mask):
        robot = env.scene["robot"]

        steps = self.cnt[env_ids].clamp(min=1.0)
        stability_mean = (self.sum_stability[env_ids] / steps).nan_to_num(0.0)
        v_mae_mean = (self.sum_v_mae[env_ids] / steps).nan_to_num(0.0)
        tilt_rms = torch.sqrt((self.sum_tilt_sq[env_ids] / steps).nan_to_num(0.0))

        slip_ratio = torch.zeros_like(steps)
        mask_has_contact = self.contact_steps[env_ids] > 0
        slip_ratio[mask_has_contact] = (
            self.slip_cnt[env_ids][mask_has_contact]
            / self.contact_steps[env_ids][mask_has_contact]
        ).clamp(0, 1)

        x_now = robot.data.root_pos_w[env_ids, 0]
        x0 = torch.where(torch.isnan(self.x0[env_ids]), x_now, self.x0[env_ids])
        forward_prog = x_now - x0

        success = (
            (~early_terminated_mask)
            & (v_mae_mean <= self.cfg.success_v_mae_max)
            & (forward_prog >= self.cfg.success_prog_min)
        )

        # env.extras 로그 (평균)
        env.extras["eval/StabilityScore_mean"] = float(stability_mean.mean().item())
        env.extras["eval/SuccessRate"] = float(success.float().mean().item())
        env.extras["eval/V_MAE_mean"] = float(v_mae_mean.mean().item())
        env.extras["eval/Tilt_RMS_mean_rad"] = float(tilt_rms.mean().item())
        env.extras["eval/Slip_Ratio_mean"] = float(slip_ratio.mean().item())
        env.extras["eval/Fwd_Progress_mean_m"] = float(forward_prog.mean().item())

        return {
            "stability_mean": stability_mean,
            "v_mae_mean": v_mae_mean,
            "tilt_rms": tilt_rms,
            "slip_ratio": slip_ratio,
            "forward_prog": forward_prog,
            "success": success,
        }


# main env
class Go2PiperVisionEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: Go2PiperVisionEnvCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._global_step = 0

        # Vision CNN (추론 전용)
        self.vision = TerrainCNN().to(self.device)
        self.vision.eval()

        # 온라인 학습 설정
        self._cnn_loss = nn.SmoothL1Loss()  # Huber 계열
        self._cnn_opt = torch.optim.AdamW(
            self.vision.parameters(), lr=5e-4, weight_decay=1e-4
        )
        self._cnn_train_every_reset = True  # 리셋 때 1 step 학습
        # Vision 캐시
        self._vision_pred = None
        self._vision_conf = None

        # 에피소드 메트릭 수집기
        self.eval_meter = EpisodeMeter(self.num_envs, self.device)


def _train_cnn_once(self, x, y):
    """한 번의 미니배치 회귀 업데이트 (x: (N,3,H,W), y: (N,))"""
    x = x.to(self.device, dtype=torch.float32)
    y = y.to(self.device, dtype=torch.float32)

    # ---------- (A) 우선 autograd 경로 시도 ----------
    try:
        with torch.enable_grad():
            # 파라미터가 얼어있으면 풀기
            for p in self.vision.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)

            self.vision.train()
            pred = self.vision(x)  # ← 반드시 forward(), predict() 금지
            if pred.requires_grad:
                loss = self._cnn_loss(pred, y)
                self._cnn_opt.zero_grad(set_to_none=True)
                loss.backward()
                self._cnn_opt.step()
                self.vision.eval()
                self.extras["vision/online_reg_loss"] = float(
                    loss.detach().mean().item()
                )
                return  # 정상 학습 완료 → 종료
    except Exception as e:
        # autograd 경로에서 문제가 나면 수동 경로로 폴백
        print("[WARN] autograd train failed, fallback to manual head SGD:", e)

    # ---------- (B) 폴백: fc2(헤드)만 수동 SGD 업데이트 ----------
    # conv/FC1은 특징 추출기로만 사용 (고정), 마지막 fc2만 수동으로 갱신
    # z: (N,128) 특징, y_pred: (N,), w: (128,), b: ()
    with torch.no_grad():
        self.vision.eval()
        z = F.relu(self.vision.conv1(x))
        z = F.relu(self.vision.conv2(z))
        z = F.relu(self.vision.conv3(z))
        z = torch.flatten(z, 1)
        z = F.relu(self.vision.fc1(z))  # (N,128)

        w = self.vision.fc2.weight.squeeze(0)  # (128,)
        b = self.vision.fc2.bias.squeeze(0)  # ()

        y_pred = z @ w + b  # (N,)
        e = y_pred - y  # (N,)

        # 평균 그라디언트 (MSE의 SGD): dL/dw = (e * z).mean(dim=0), dL/db = e.mean()
        grad_w = (e.unsqueeze(1) * z).mean(dim=0)  # (128,)
        grad_b = e.mean()  # ()

        eta = 5e-4  # 헤드 전용 학습률 (필요시 살짝 키워도 됨)
        self.vision.fc2.weight -= eta * grad_w.unsqueeze(0)
        self.vision.fc2.bias -= eta * grad_b

        self.extras["vision/online_reg_loss"] = float((e.pow(2).mean()).item())

    # -------- Vision 유틸 --------
    def _vision_smoke(self):
        x = preprocess_rgb(self, "ee_cam", 96)
        self.extras["vision/shape_ok"] = float(x.shape[-1] == 96)
        self.extras["vision/device_cuda"] = float(x.is_cuda)

    @torch.no_grad()
    def _vision_predict(self):
        x = preprocess_rgb(self, "ee_cam", 96)
        # 회귀: predict() → (y, conf)
        y_pred, conf = self.vision.predict(x)  # y_pred: (N,) float
        self._vision_pred = y_pred
        self._vision_conf = conf
        self.extras["vision/pred_mean"] = float(y_pred.float().mean().item())
        self.extras["vision/conf_mean"] = float(conf.mean().item())

    # -------- Stability (겹침 제거 버전) --------
    def _compute_stability_score(self):
        """
        자세/지지 중심 안정도 [0,1]:
          - s_height: |z - z*|
          - s_omega: |ω_x,y|
          - s_contact: 접지 개수(2~3개 선호)
          - s_support: support polygon(간단히 bbox 근사) 내부 여유 margin
          - Slip, V_MAE, Progress는 포함하지 않음 (중복 방지).
        """
        device = self.device
        robot = self.scene["robot"]
        N = robot.data.root_pos_w.shape[0]

        # 높이 안정성
        z = robot.data.root_pos_w[:, 2]
        target_h, tol_h = 0.42, 0.08
        s_height = torch.exp(-torch.abs(z - target_h) / tol_h).clamp(0, 1)

        # 기울기/진동 proxy: 각속도(x,y)
        ang_xy = robot.data.root_ang_vel_w[:, :2].norm(dim=-1)
        s_omega = torch.exp(-ang_xy / 1.0).clamp(0, 1)

        # 발 인덱스
        names = robot.data.body_names
        foot_names = ["FL_foot", "FR_foot", "HL_foot", "HR_foot"]
        foot_ids = [names.index(n) for n in foot_names if n in names]

        # 접지 개수 안정성 (2~3개 선호)
        if len(foot_ids) > 0 and "contact_forces" in self.scene.sensors:
            cs: ContactSensor = self.scene.sensors["contact_forces"]
            try:
                netF = cs.data.net_forces_w[:, foot_ids, :]  # (N,4,3)
                contacts = netF.norm(dim=-1) > 1.0  # (N,4)
            except Exception:
                netF = cs.data.net_forces_w_history[:, -1, foot_ids, :]
                contacts = netF.norm(dim=-1) > 1.0
            contact_cnt = contacts.sum(dim=1).float()
            s_contact = torch.exp(-torch.abs(contact_cnt - 2.5) * 0.7).clamp(0, 1)
        else:
            s_contact = torch.ones(N, device=device)

        # Support polygon margin (bbox 근사)
        #   - 발의 x,y 최소/최대로 bbox 생성
        #   - CoM 투영(여기서는 root x,y)과 bbox 거리로 margin 계산
        #   - margin > 0 (안) / < 0 (밖)
        if len(foot_ids) > 0:
            feet_xy = robot.data.body_pos_w[:, foot_ids, :2]  # (N,4,2)
            x_min, _ = feet_xy[:, :, 0].min(dim=1)
            x_max, _ = feet_xy[:, :, 0].max(dim=1)
            y_min, _ = feet_xy[:, :, 1].min(dim=1)
            y_max, _ = feet_xy[:, :, 1].max(dim=1)
            com_xy = robot.data.root_pos_w[:, :2]  # (N,2)

            margin_x = torch.minimum(x_max - com_xy[:, 0], com_xy[:, 0] - x_min)
            margin_y = torch.minimum(y_max - com_xy[:, 1], com_xy[:, 1] - y_min)
            margin = torch.minimum(margin_x, margin_y)  # (N,)

            tau_m = 0.10  # 여유 정규화 스케일
            s_support = torch.sigmoid(margin / tau_m)
        else:
            s_support = torch.ones(N, device=device)

        # 종합 안정도 (가중 기하평균)
        score = (
            (s_height.clamp(0, 1) ** 0.35)
            * (s_omega.clamp(0, 1) ** 0.35)
            * (s_contact.clamp(0, 1) ** 0.15)
            * (s_support.clamp(0, 1) ** 0.15)
        ).clamp(0, 1)

        if not hasattr(self, "metrics"):
            self.metrics = {}
        self.metrics["stability_score"] = score
        self.extras["stability/mean"] = float(score.mean().item())

    # -------- 타이밍 로그 --------
    def _log_timing_constants(self):
        physics_dt = (
            float(self.sim.get_physics_dt())
            if hasattr(self.sim, "get_physics_dt")
            else 1.0 / 120.0
        )
        decimation = int(
            getattr(self.cfg.sim, "decimation", getattr(self.cfg, "decimation", 4))
        )
        episode_length_s = float(getattr(self.cfg, "episode_length_s", 20.0))
        control_dt = physics_dt * decimation
        steps_per_episode = round(episode_length_s / control_dt)
        self.extras["timing/physics_dt_s"] = physics_dt
        self.extras["timing/control_dt_s"] = control_dt
        self.extras["timing/steps_per_episode"] = float(steps_per_episode)

    # -------- 스텝 훅 --------
    def post_physics_step(self):
        super().post_physics_step()
        if not hasattr(self, "_timing_logged"):
            self._log_timing_constants()
            self._timing_logged = True
        self._global_step += 1

        # 안정도 갱신(중복 제거 정의)
        self._compute_stability_score()

        # 50step 헬스체크
        if self._global_step % 50 == 0:
            self._vision_smoke()

        # 에피소드 메트릭 누적 (최종 5지표)
        self.eval_meter.step_accumulate(self)

    # -------- 리셋 훅 --------
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() > 0:
            # 조기 종료 마스크 (실제 terminated/truncated 마스크로 교체 가능)
            early_terminated = torch.zeros_like(
                env_ids, dtype=torch.bool, device=self.device
            )

            # 에피소드 통계 집계 & 로그 (최종 5지표)
            self.eval_meter.finalize_episode(self, env_ids, early_terminated)

            # 다음 에피소드용 버퍼 리셋
            self.eval_meter.reset_buffers(env_ids)

            x = preprocess_rgb(self, "ee_cam", 96)
            y = self.scene.terrain.terrain_levels.clone().to(self.device).float()

            if self._cnn_train_every_reset:
                with torch.enable_grad():
                    self._train_cnn_once(x, y)

            # Vision inference 캐시 갱신
            self._vision_smoke()
            self._vision_predict()

        super()._reset_idx(env_ids)
