import torch
import numpy as np
import math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


class HexapodSandTrainEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # RFT相关初始化
        self.rft_generic_M = torch.tensor([0.206, 0.169, 0.212, 0.358, 0.055, -0.124, 0.253, 0.007, 0.088], 
                                        device=self.device)
        self.sand_zeta = 0.5  # 适度增大阻力系数
        self.foot_area = 0.01  # 更合理的足端面积
        self.foot_bottom_offset = 0.06  # 增大偏移，确保能计算出入沙深度
        self.min_sink_depth = 0.001  # 给一个小的最小入沙深度，确保有接触就有力
        self.contact_force_threshold = 1.0
        self._debug_print_count = 0
        self.foot_body_names = ["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"]
        
        # 延迟初始化的标志
        self._initialized_feet = False
    
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
        """应用RFT力"""
        robot = self.scene["robot"]
        contact_sensor = self.scene["contact_forces"]
        foot_positions = robot.data.body_pos_w[:, self.foot_indices, :]
        foot_vel = robot.data.body_lin_vel_w[:, self.foot_indices, :]
        contact_history = contact_sensor.data.net_forces_w_history[:, :, self.foot_indices, :]
        contact_force_mag = contact_history.norm(dim=-1).max(dim=1)[0]
        in_sand_mask = contact_force_mag > self.contact_force_threshold

        # 用“脚底偏移”近似脚底世界高度；有接触力时就认为在沙中
        foot_bottom_height = foot_positions[..., 2] - self.foot_bottom_offset
        # 即使脚底高度略高于0，只要有接触，就给一个基础深度
        depth_from_height = torch.clamp(-foot_bottom_height, min=0.0)
        # 对于有接触但深度为0的情况，给一个最小深度
        base_depth = torch.where(
            in_sand_mask,
            self.min_sink_depth + torch.clamp(-foot_bottom_height * 0.5, min=0.0),
            torch.zeros_like(foot_bottom_height),
        )
        depth = torch.maximum(depth_from_height, base_depth)

        forces = torch.zeros((self.num_envs, len(self.foot_indices), 3), device=self.device)
        torques = torch.zeros_like(forces)

        downward_speed = torch.clamp(-foot_vel[..., 2], min=0.0)
        upward_speed = torch.clamp(foot_vel[..., 2], min=0.0)
        
        # 垂直力：向下运动时提供支撑/阻力，向上运动时提供较小阻力
        vertical_force = self.sand_zeta * self.foot_area * (depth * 50.0 + downward_speed * 10.0)
        # 向上运动时也有轻微阻力
        vertical_force -= self.sand_zeta * self.foot_area * upward_speed * 2.0
        
        forces[..., 2] = torch.where(
            in_sand_mask,
            vertical_force,
            torch.zeros_like(vertical_force),
        )
        # 水平阻力：更合理的阻尼系数
        forces[..., 0:2] = torch.where(
            in_sand_mask.unsqueeze(-1),
            -self.sand_zeta * 20.0 * foot_vel[..., 0:2],
            torch.zeros_like(foot_vel[..., 0:2]),
        )

        robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=self.foot_indices, is_global=True)

        self._debug_print_count += 1
        if self._debug_print_count % 200 == 0:
            max_depth = float(depth.max().item())
            max_fz = float(forces[..., 2].max().item())
            max_contact = float(contact_force_mag.max().item())
            min_foot_bottom_h = float(foot_bottom_height.min().item())
            print(
                f"[DEBUG RFT]: step={self._debug_print_count} "
                f"max_contact={max_contact:.3f} min_foot_bottom_h={min_foot_bottom_h:.4f} "
                f"max_depth={max_depth:.4f} max_fz={max_fz:.3f}"
            )

        return forces
    
    def step(self, action: torch.Tensor):
        if not self._initialized_feet:
            self._init_feet()
        self._apply_rft()
        return super().step(action)
