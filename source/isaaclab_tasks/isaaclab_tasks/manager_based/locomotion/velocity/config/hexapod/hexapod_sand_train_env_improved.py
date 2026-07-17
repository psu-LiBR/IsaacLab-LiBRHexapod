
import torch
import numpy as np
import math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


class HexapodSandImprovedTrainEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # RFT相关初始化
        self.rft_generic_M = torch.tensor([0.206, 0.169, 0.212, 0.358, 0.055, -0.124, 0.253, 0.007, 0.088], 
                                        device=self.device)
        self.sand_zeta = 3.5  # 阻力系数：回调一些，找平衡点
        self.foot_area = 0.02  # 足端面积：保持
        self.foot_bottom_offset = 0.0  # 脚底位置偏移：保持
        self.min_sink_depth = 0.001  # 最小入沙深度：保持
        self.contact_force_threshold = 1.0  # 接触力阈值：保持
        self.vertical_depth_gain = 7000.0  # 垂向支撑增益：回调
        self.vertical_speed_gain = 150.0  # 垂向速度增益：回调
        self.horizontal_depth_gain = 5000.0  # 水平阻力增益：回调
        self.horizontal_speed_gain = 100.0  # 水平速度增益：回调
        self.posture_coupling_gain = 0.2  # 姿态耦合增益：保持
        self.upward_relief_gain = 15.0  # 向上卸载增益：回调
        self.force_limit = 40.0  # 力限制：回调
        self._debug_print_count = 0
        self.foot_body_names = ["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"]
        
        # 延迟初始化的标志
        self._initialized_feet = False

    def _quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vectors by quaternions in wxyz convention."""
        quat_xyz = quat[..., 1:]
        quat_w = quat[..., :1]
        twice_cross = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
        return vec + quat_w * twice_cross + torch.cross(quat_xyz, twice_cross, dim=-1)

    def _compute_rft_coefficients(
        self,
        foot_normal: torch.Tensor,
        foot_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map leg orientation and intrusion direction into simplified terradynamics coefficients."""
        eps = 1.0e-6
        tangential_speed = torch.linalg.norm(foot_vel[..., :2], dim=-1)
        intrusion_angle = torch.atan2(-foot_vel[..., 2], tangential_speed + eps)
        attack_angle = torch.acos(torch.clamp(torch.abs(foot_normal[..., 2]), 0.0, 1.0))
        phase_angle = intrusion_angle - attack_angle

        m = self.rft_generic_M
        vertical_coeff = m[0] + m[1] * torch.cos(attack_angle) + m[2] * torch.sin(attack_angle)
        drag_coeff = m[3] + m[4] * torch.cos(intrusion_angle) + m[5] * torch.sin(intrusion_angle)
        coupling_coeff = m[6] + m[7] * torch.cos(phase_angle) + m[8] * torch.sin(phase_angle)

        return (
            torch.clamp(torch.abs(vertical_coeff), min=0.05),
            torch.clamp(torch.abs(drag_coeff), min=0.05),
            torch.clamp(torch.abs(coupling_coeff), min=0.05),
        )
    
    def _init_feet(self):
        if self._initialized_feet:
            return

        robot = self.scene["robot"]
        foot_indices, resolved_names = robot.find_bodies(self.foot_body_names)
        if len(foot_indices) != len(self.foot_body_names):
            raise RuntimeError(
                f"足端 body 解析失败，期望 {self.foot_body_names}，实际解析到 {resolved_names} / {foot_indices}"
            )

        self.foot_indices = foot_indices.tolist() if hasattr(foot_indices, "tolist") else list(foot_indices)
        self.foot_resolved_names = list(resolved_names)
        print(f"[INFO]: 真实足端 body 名称: {self.foot_resolved_names}")
        print(f"[INFO]: 真实足端 body 索引: {self.foot_indices}")
        self._initialized_feet = True
    
    def _apply_rft(self):
        """应用改进版 RFT 力：使用姿态角、侵入角和 rft_generic_M 系数矩阵。"""
        robot = self.scene["robot"]
        contact_sensor = self.scene["contact_forces"]
        foot_positions = robot.data.body_pos_w[:, self.foot_indices, :]
        foot_vel = robot.data.body_lin_vel_w[:, self.foot_indices, :]
        foot_quat = robot.data.body_quat_w[:, self.foot_indices, :]
        contact_history = contact_sensor.data.net_forces_w_history[:, :, self.foot_indices, :]
        contact_force_mag = contact_history.norm(dim=-1).max(dim=1)[0]
        in_sand_mask = contact_force_mag > self.contact_force_threshold

        # 用“脚底偏移”近似脚底世界高度；有接触力时就认为在沙中
        foot_bottom_height = foot_positions[..., 2] - self.foot_bottom_offset
        # 深度计算：直接使用 -foot_bottom_height，只要脚在地面以下就算入沙
        # 这样能更准确地反映真实入沙深度
        depth = torch.clamp(-foot_bottom_height, min=0.0)
        # 对于有接触但计算深度为0的情况，给一个最小深度确保有力
        depth = torch.where(
            torch.logical_and(in_sand_mask, depth < self.min_sink_depth),
            self.min_sink_depth,
            depth
        )

        forces = torch.zeros((self.num_envs, len(self.foot_indices), 3), device=self.device)
        torques = torch.zeros_like(forces)

        local_normal = torch.zeros_like(foot_vel)
        local_normal[..., 2] = 1.0
        foot_normal = self._quat_rotate(foot_quat, local_normal)
        foot_normal = foot_normal / (torch.linalg.norm(foot_normal, dim=-1, keepdim=True) + 1.0e-6)
        vertical_coeff, drag_coeff, coupling_coeff = self._compute_rft_coefficients(foot_normal, foot_vel)

        downward_speed = torch.clamp(-foot_vel[..., 2], min=0.0)
        upward_speed = torch.clamp(foot_vel[..., 2], min=0.0)
        tangential_vel = foot_vel[..., :2]
        tangential_speed = torch.linalg.norm(tangential_vel, dim=-1)
        tangential_dir = tangential_vel / (tangential_speed.unsqueeze(-1) + 1.0e-6)
        normal_xy = foot_normal[..., :2]
        normal_xy = normal_xy / (torch.linalg.norm(normal_xy, dim=-1, keepdim=True) + 1.0e-6)

        vertical_force = self.sand_zeta * self.foot_area * (
            depth * self.vertical_depth_gain * vertical_coeff
            + downward_speed * self.vertical_speed_gain * (0.5 + coupling_coeff)
            - upward_speed * self.upward_relief_gain * drag_coeff
        )

        forces[..., 2] = torch.where(
            in_sand_mask,
            vertical_force,
            torch.zeros_like(vertical_force),
        )

        horizontal_drag_mag = self.sand_zeta * self.foot_area * (
            depth * self.horizontal_depth_gain * drag_coeff
            + tangential_speed * self.horizontal_speed_gain * (0.5 + coupling_coeff)
        )
        posture_coupling = (
            self.sand_zeta
            * self.foot_area
            * depth.unsqueeze(-1)
            * self.posture_coupling_gain
            * coupling_coeff.unsqueeze(-1)
            * normal_xy
        )
        horizontal_force = -tangential_dir * horizontal_drag_mag.unsqueeze(-1) + posture_coupling
        forces[..., 0:2] = torch.where(
            in_sand_mask.unsqueeze(-1),
            horizontal_force,
            torch.zeros_like(horizontal_force),
        )
        forces = torch.clamp(forces, min=-self.force_limit, max=self.force_limit)

        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=self.foot_indices,
            is_global=True,
        )

        self._debug_print_count += 1
        if self._debug_print_count % 200 == 0:
            max_depth = float(depth.max().item())
            max_fz = float(forces[..., 2].max().item())
            max_contact = float(contact_force_mag.max().item())
            min_foot_bottom_h = float(foot_bottom_height.min().item())
            mean_vertical_coeff = float(vertical_coeff.mean().item())
            mean_drag_coeff = float(drag_coeff.mean().item())
            print(
                f"[DEBUG RFT]: step={self._debug_print_count} "
                f"max_contact={max_contact:.3f} min_foot_bottom_h={min_foot_bottom_h:.4f} "
                f"max_depth={max_depth:.4f} max_fz={max_fz:.3f} "
                f"cv={mean_vertical_coeff:.3f} cd={mean_drag_coeff:.3f}"
            )

        return forces
    
    def step(self, action: torch.Tensor):
        if not self._initialized_feet:
            self._init_feet()
        self._apply_rft()
        return super().step(action)
