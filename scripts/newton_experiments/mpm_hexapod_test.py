# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-License-Identifier: BSD-3-Clause
"""Standalone Newton simulation: LiBR Hexapod walking over MPM granular sand.

This is NOT part of the Isaac Lab / Isaac Sim training pipeline. It loads the
hexapod's own USD asset directly through Newton's ModelBuilder and drives it
open-loop from a recorded gait CSV (the same "Sim Gaits" CSVs used by
playReal.py), coupled one-way to Newton's implicit-MPM granular solver. The
purpose is to visually sanity-check how the robot interacts with granular
terrain before investing in wiring MPM support into Isaac Sim's
`isaacsim.physics.newton` bridge extension (which today only supports the
xpbd/mujoco solvers -- see newton_stage.py's `_get_solver`).

Requires the Isaac Sim 6.0 / Newton environment (pip env with `newton`,
`newton-actuators`, `isaacsim` 6.0 installed), e.g.:

    D:/uv_envs/env_isaaclab3/Scripts/python.exe scripts/newton_experiments/mpm_hexapod_test.py \
        --viewer usd --output-path hexapod_mpm.usd --num-frames 300

Start with --viewer usd (writes a USD file, played back later in usdview or
Isaac Sim) rather than --viewer gl: this machine's GPU driver is known to
crash Isaac Sim's RTX rendering path (see CLAUDE.md's PyVista offline
rendering section). Newton's GL viewer is a separate plain-OpenGL path and
may well be fine, but that hasn't been verified on this machine.
"""

from __future__ import annotations

import csv
import os

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from pxr import Usd, UsdGeom, UsdPhysics

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HEXAPOD_USD_PATH = os.path.join(_REPO_ROOT, "hexapod-assets", "USD", "Hexapod_Flattened.usd")
DEFAULT_GAIT_CSV = os.path.join(_REPO_ROOT, "hexapod-assets", "Sim Gaits", "forward3_lleg30_amp65_sim.csv")


def _first_mesh_descendant(prim: Usd.Prim) -> Usd.Prim | None:
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            return child
        found = _first_mesh_descendant(child)
        if found is not None:
            return found
    return None


def _load_deinstanced_stage(usd_path: str) -> Usd.Stage:
    """Open the hexapod USD, fixed up so Newton's add_usd can find collision geometry.

    Two problems in Hexapod_Flattened.usd, neither of which affects PhysX/Isaac
    Sim (which is more lenient about resolving this), but both of which mean
    Newton finds zero collision shapes on every body:

    1. Each body's visual/collision mesh is stored via native USD instancing
       (18 prototypes). Newton's add_usd walks the composed scenegraph
       directly and doesn't resolve through instance prototypes, so the
       ".../collisions" and ".../visuals" prims appear to have no children at
       all. Fix: de-instance every instance prim in-memory (no new file
       written) so the real mesh is exposed as an ordinary composed child.
    2. Even after de-instancing, `PhysicsCollisionAPI` is applied to the
       ".../collisions" wrapper Xform itself, not to the actual Mesh prim
       nested under it (".../collisions/<Body>/mesh") -- Newton only
       recognizes CollisionAPI applied directly to a Gprim. Fix: apply
       CollisionAPI directly to each ".../collisions" wrapper's real Mesh
       descendant. Deliberately plain triangle-mesh collision (no
       MeshCollisionAPI/convexHull approximation): Newton's implicit-MPM
       collider builder only supports GeoType.MESH/PLANE/SPHERE/CAPSULE/
       CYLINDER/CONE/BOX, not CONVEX_MESH, so carrying over the wrapper's
       authored `physics:approximation = convexHull` would make the MPM
       solver raise `NotImplementedError: Shape type 10 not supported`.
    """
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)

    for prim in stage.Traverse():
        if prim.GetName() == "collisions" and prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_prim = _first_mesh_descendant(prim)
            if mesh_prim is None:
                continue
            UsdPhysics.CollisionAPI.Apply(mesh_prim)

    return stage


# Matches CLAUDE.md's documented Sim DOF order / MotionReference joint ordering,
# and the column order of the "Sim Gaits" CSVs.
JOINT_NAMES = [
    "BackLink_Joint",
    "FrontLink_Joint",
    "MiddleLeft_Joint",
    "MiddleRight_Joint",
    "BackLeft_Joint",
    "BackRight_Joint",
    "FrontLeft_Joint",
    "FrontRight_Joint",
]
SPINE_JOINTS = {"BackLink_Joint", "FrontLink_Joint"}
# Body/link names -- unaffected by the joint-prim rename in Hexapod_Flattened.usd
# (only the joint prims were renamed with a "_Joint" suffix; the driven bodies kept
# their original names to disambiguate them from the joints for omni.physx.tensors).
LEG_LINK_NAMES = {"BackLeft", "BackRight", "FrontLeft", "FrontRight", "MiddleLeft", "MiddleRight"}

# PD gains matching HEXAPOD_CFG.actuators in
# source/isaaclab_assets/isaaclab_assets/robots/hexapod.py. Newton's
# joint_target_ke/kd are the analogous PD spring/damper terms, but unit
# conventions between PhysX's implicit actuator and Newton's solvers aren't
# guaranteed to match 1:1 -- treat these as a starting point, not calibrated.
SPINE_KE, SPINE_KD = 40.0, 0.4
LEG_KE, LEG_KD = 80.0, 0.9

# Matches HEXAPOD_CFG.init_state.joint_pos.
INIT_JOINT_POS = {
    "FrontLink_Joint": 0.0,
    "BackLink_Joint": 0.0,
    "MiddleLeft_Joint": -0.47,
    "MiddleRight_Joint": -0.47,
    "BackLeft_Joint": -0.47,
    "BackRight_Joint": -0.47,
    "FrontLeft_Joint": -0.47,
    "FrontRight_Joint": -0.47,
}


def _load_gait_csv(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        rows = [[float(v) for v in row] for row in csv.reader(f) if row]
    if not rows:
        raise ValueError(f"Gait CSV '{path}' has no rows.")
    arr = np.array(rows, dtype=np.float32)
    if arr.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"Gait CSV '{path}' has {arr.shape[1]} columns, expected {len(JOINT_NAMES)}.")
    return arr


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.mu = 0.5
        builder.default_joint_cfg.armature = 0.01

        hexapod_stage = _load_deinstanced_stage(args.hexapod_usd)
        builder.add_usd(
            hexapod_stage,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
            floating=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )

        # Match joint names to builder indices. builder.joint_label includes
        # the floating base's free joint as well as the 8 revolute joints.
        joint_idx: dict[str, int] = {}
        for name in JOINT_NAMES:
            for i, lbl in enumerate(builder.joint_label):
                if lbl == name or lbl.endswith(f"/{name}"):
                    joint_idx[name] = i
                    break
        missing = [n for n in JOINT_NAMES if n not in joint_idx]
        if missing:
            raise RuntimeError(f"Could not find joints {missing} in USD. Found joint labels: {builder.joint_label}")

        # Initial pose + PD gains, via each joint's own q/qd start offsets
        # (robust to however add_usd orders/sizes the floating base's dofs).
        for name, value in INIT_JOINT_POS.items():
            i = joint_idx[name]
            builder.joint_q[builder.joint_q_start[i]] = value
            ke, kd = (SPINE_KE, SPINE_KD) if name in SPINE_JOINTS else (LEG_KE, LEG_KD)
            builder.joint_target_ke[builder.joint_qd_start[i]] = ke
            builder.joint_target_kd[builder.joint_qd_start[i]] = kd

        # Only let the six leg links collide with sand particles (spine/body
        # links stay collision-free against particles), mirroring the
        # SHANK-only filter in newton's example_mpm_anymal.py.
        for body in range(builder.body_count):
            label = builder.body_label[body]
            if not any(leg in label for leg in LEG_LINK_NAMES):
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        # Register MPM custom attributes before adding particles.
        SolverImplicitMPM.register_custom_attributes(builder)

        particle_lo = np.array(args.sand_lo, dtype=np.float64)
        particle_hi = np.array(args.sand_hi, dtype=np.float64)
        particle_res = np.array(
            np.ceil(args.particles_per_cell * (particle_hi - particle_lo) / args.voxel_size), dtype=int
        )
        cell_size = (particle_hi - particle_lo) / particle_res
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_size) * args.sand_density)
        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(particle_res[0]) + 1,
            dim_y=int(particle_res[1]) + 1,
            dim_z=int(particle_res[2]) + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )

        builder.add_ground_plane()

        # Save qd-space indices before finalize (builder attrs map 1:1 to Model attrs).
        self.joint_qd_indices = {name: builder.joint_qd_start[joint_idx[name]] for name in JOINT_NAMES}

        self.model = builder.finalize()

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = args.voxel_size
        mpm_options.tolerance = args.tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = args.grid_type
        mpm_options.grid_padding = 50 if args.grid_type == "fixed" else 0
        mpm_options.max_active_cell_count = (1 << 15) if args.grid_type == "fixed" else -1
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "backward"

        self.solver = newton.solvers.SolverXPBD(self.model)
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Treat the robot as a kinematic collider for the (one-way-coupled)
        # sand solver: the robot deforms the sand, but sand contact forces
        # are not fed back into the rigid-body solver's dynamics.
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        self.control = self.model.control()

        self.gait = _load_gait_csv(args.gait_csv)
        self.gait_dt = args.gait_dt
        self.gait_duration = self.gait.shape[0] * self.gait_dt

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def _gait_targets(self, t: float) -> np.ndarray:
        phase_t = t % self.gait_duration
        row = phase_t / self.gait_dt
        i0 = int(row) % self.gait.shape[0]
        i1 = (i0 + 1) % self.gait.shape[0]
        frac = row - int(row)
        return (1.0 - frac) * self.gait[i0] + frac * self.gait[i1]

    def apply_control(self):
        targets = self._gait_targets(self.sim_time)
        target_full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for name, value in zip(JOINT_NAMES, targets):
            target_full[self.joint_qd_indices[name]] = value
        wp.copy(self.control.joint_target_pos, wp.array(target_full, dtype=wp.float32, device=self.model.device))

    def simulate_robot(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        self.apply_control()
        self.simulate_robot()
        self.simulate_sand()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--hexapod-usd", type=str, default=HEXAPOD_USD_PATH)
        parser.add_argument("--gait-csv", type=str, default=DEFAULT_GAIT_CSV)
        parser.add_argument(
            "--gait-dt",
            type=float,
            default=0.01,
            help=(
                "Seconds per row in --gait-csv. Default assumes the 50-row "
                "forward3_lleg30_amp65_sim.csv spans one 0.5s gait cycle "
                "(MIMIC_GAIT_PERIOD) -- adjust if using a different CSV."
            ),
        )
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.015)
        parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
        parser.add_argument("--grid-type", "-gt", choices=["sparse", "dense", "fixed"], default="sparse")
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument("--sand-density", type=float, default=1600.0, help="kg/m^3, loose dry sand.")
        parser.add_argument("--sand-lo", type=float, nargs=3, default=[-0.3, -0.3, 0.0], metavar=("X", "Y", "Z"))
        parser.add_argument("--sand-hi", type=float, nargs=3, default=[1.2, 0.3, 0.08], metavar=("X", "Y", "Z"))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
