# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-License-Identifier: BSD-3-Clause
"""Standalone Newton simulation: LiBR Hexapod, TWO-WAY coupled to MPM granular sand.

Unlike mpm_hexapod_test.py (one-way: the robot pushes sand, sand never pushes
back), this feeds MPM contact impulses back into the rigid-body solver as
per-body forces/torques, so the robot's own dynamics -- sinking, slipping,
extra torque from soft terrain -- are actually informed by the granular
terrain. Intended for collecting comparison data against real hardware runs,
not just a visual check.

Coupling scheme mirrors newton/examples/mpm/example_mpm_twoway_coupling.py
(a validated upstream reference), adapted to the hexapod's USD asset and its
open-loop gait CSV instead of falling boxes:

  - The rigid model (hexapod + ground plane) and the sand model (particles
    only) are two SEPARATE newton.Model instances. `setup_collider(model=...)`
    reads the rigid model's real body mass/inertia (not zeroed/kinematic), so
    contact response is dynamically consistent, not one-way kinematic.
  - Impulses collected from the sand solver are converted to forces/torques
    at each body's center of mass and written into `state.body_f` every rigid
    substep (`compute_body_forces`).
  - Before the sand solver steps again, the force it applied last frame is
    subtracted back out of the collider's velocity (`subtract_body_force`) --
    required so the MPM solver's complementarity contact solve sees the
    rigid body's *unconstrained* velocity, not one already reactively
    adjusted by last frame's sand force (would otherwise double-count).
  - The robot's leg links are excluded from colliding with the rigid ground
    plane (add_shape_collision_filter_pair), since within the sand pit the
    sand itself is the only thing that should be supporting the robot's
    weight -- leaving the rigid floor connected would silently short-circuit
    the entire coupling (the legs would just stand on the invisible floor
    regardless of what the sand is doing).

Requires the Isaac Sim 6.0 / Newton environment, e.g.:

    D:/uv_envs/env_isaaclab3/Scripts/python.exe scripts/newton_experiments/mpm_hexapod_twoway.py \
        --viewer usd --output-path hexapod_mpm_twoway.usd --num-frames 300 --log-csv hexapod_mpm_dynamics.csv

Logged CSV columns (one row per rendered frame, quaternions in Warp's
XYZW/scalar-last convention): frame, sim_time, base_pos_{x,y,z},
base_quat_{x,y,z,w}, base_linvel_{x,y,z}, base_angvel_{x,y,z}, then per-joint
position and gait-target columns for all 8 joints, then sand_force_{x,y,z}
and sand_torque_{x,y,z} (summed over all leg bodies, world frame).
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

SPINE_KE, SPINE_KD = 40.0, 0.4
LEG_KE, LEG_KD = 80.0, 0.9

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


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Sum sand impulses per MPM collider node into per-body force/torque at the body COM."""
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return
        f_world = collider_impulses[i] / dt
        x_wb = body_q[body_index]
        x_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(x_wb, x_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Undo last frame's applied sand force so MPM sees the collider's unconstrained velocity."""
    body_id = wp.tid()
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.viewer = viewer

        # ---------------- rigid-body model (hexapod + ground plane) ----------------
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

        joint_idx: dict[str, int] = {}
        body_idx: dict[str, int] = {}
        for i, lbl in enumerate(builder.joint_label):
            for name in JOINT_NAMES:
                if name not in joint_idx and (lbl == name or lbl.endswith(f"/{name}")):
                    joint_idx[name] = i
        for b, lbl in enumerate(builder.body_label):
            for name in LEG_LINK_NAMES | {"CenterLink"}:
                if name not in body_idx and (lbl == name or lbl.endswith(f"/{name}")):
                    body_idx[name] = b
        missing = [n for n in JOINT_NAMES if n not in joint_idx]
        if missing:
            raise RuntimeError(f"Could not find joints {missing} in USD. Found labels: {builder.joint_label}")
        if "CenterLink" not in body_idx:
            raise RuntimeError(f"Could not find CenterLink body in USD. Found labels: {builder.body_label}")

        for name, value in INIT_JOINT_POS.items():
            i = joint_idx[name]
            builder.joint_q[builder.joint_q_start[i]] = value
            ke, kd = (SPINE_KE, SPINE_KD) if name in SPINE_JOINTS else (LEG_KE, LEG_KD)
            builder.joint_target_ke[builder.joint_qd_start[i]] = ke
            builder.joint_target_kd[builder.joint_qd_start[i]] = kd

        ground_shape = builder.add_ground_plane()

        # Only leg links interact with sand particles; spine/body links stay clear.
        # Leg links are also excluded from the rigid ground plane -- within the
        # sand pit, sand-derived forces are the *only* thing holding the robot
        # up. Leaving ground contact enabled would let the legs stand on the
        # invisible rigid floor regardless of the sand, silently defeating the
        # coupling.
        for body in range(builder.body_count):
            label = builder.body_label[body]
            is_leg = any(leg in label for leg in LEG_LINK_NAMES)
            for shape in builder.body_shapes[body]:
                if not is_leg:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES
                else:
                    builder.add_shape_collision_filter_pair(shape, ground_shape)

        self.joint_qd_indices = {name: builder.joint_qd_start[joint_idx[name]] for name in JOINT_NAMES}
        self.joint_q_indices = {name: builder.joint_q_start[joint_idx[name]] for name in JOINT_NAMES}
        self.center_link_body = body_idx["CenterLink"]
        self.leg_body_indices = [body_idx[name] for name in LEG_LINK_NAMES if name in body_idx]

        self.model = builder.finalize()

        # ---------------- sand model (particles only) ----------------
        sand_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverImplicitMPM.register_custom_attributes(sand_builder)

        particle_lo = np.array(args.sand_lo, dtype=np.float64)
        particle_hi = np.array(args.sand_hi, dtype=np.float64)
        particle_res = np.array(
            np.ceil(args.particles_per_cell * (particle_hi - particle_lo) / args.voxel_size), dtype=int
        )
        cell_size = (particle_hi - particle_lo) / particle_res
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_size) * args.sand_density)
        sand_builder.add_particle_grid(
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
        self.sand_model = sand_builder.finalize()

        # ---------------- solvers ----------------
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = args.voxel_size
        mpm_options.tolerance = args.tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = args.grid_type
        mpm_options.grid_padding = 50 if args.grid_type == "fixed" else 0
        mpm_options.max_active_cell_count = args.max_active_cells if args.grid_type == "fixed" else -1
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "backward"

        self.mpm_solver = SolverImplicitMPM(self.sand_model, mpm_options)
        # Read colliders (and their real mass/inertia) from the rigid model --
        # dynamic response, not the one-way kinematic-collider setup.
        self.mpm_solver.setup_collider(model=self.model)

        self.solver = newton.solvers.SolverXPBD(self.model)

        # ---------------- states ----------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.sand_state_0 = self.sand_model.state()
        self.sand_state_0.body_q = wp.empty_like(self.state_0.body_q)
        self.sand_state_0.body_qd = wp.empty_like(self.state_0.body_qd)
        self.sand_state_0.body_f = wp.empty_like(self.state_0.body_f)

        self.control = self.model.control()

        # Preallocated scratch buffers for the impulse feedback loop.
        max_nodes = args.max_collider_nodes
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collider_body_id = self.mpm_solver.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)
        self._collect_collider_impulses()

        self.gait = _load_gait_csv(args.gait_csv)
        self.gait_dt = args.gait_dt
        self.gait_duration = self.gait.shape[0] * self.gait_dt

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.particle_render_colors = wp.full(
            self.sand_model.particle_count,
            value=wp.vec3(0.76, 0.70, 0.50),
            dtype=wp.vec3,
            device=self.sand_model.device,
        )

        self.log_writer = None
        self.log_file = None
        if args.log_csv:
            self.log_file = open(args.log_csv, "w", newline="")  # noqa: SIM115 -- closed in cleanup(), spans object lifetime
            self.log_writer = csv.writer(self.log_file)
            header = ["frame", "sim_time"]
            header += ["base_pos_x", "base_pos_y", "base_pos_z"]
            header += ["base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"]
            header += ["base_linvel_x", "base_linvel_y", "base_linvel_z"]
            header += ["base_angvel_x", "base_angvel_y", "base_angvel_z"]
            header += [f"joint_pos_{name}" for name in JOINT_NAMES]
            header += [f"joint_target_{name}" for name in JOINT_NAMES]
            header += ["sand_force_x", "sand_force_y", "sand_force_z"]
            header += ["sand_torque_x", "sand_torque_y", "sand_torque_z"]
            self.log_writer.writerow(header)

        self._last_targets = np.zeros(len(JOINT_NAMES), dtype=np.float32)

    def _gait_targets(self, t: float) -> np.ndarray:
        phase_t = t % self.gait_duration
        row = phase_t / self.gait_dt
        i0 = int(row) % self.gait.shape[0]
        i1 = (i0 + 1) % self.gait.shape[0]
        frac = row - int(row)
        return (1.0 - frac) * self.gait[i0] + frac * self.gait[i1]

    def apply_control(self):
        targets = self._gait_targets(self.sim_time)
        self._last_targets = targets
        target_full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for name, value in zip(JOINT_NAMES, targets):
            target_full[self.joint_qd_indices[name]] = value
        wp.copy(self.control.joint_target_pos, wp.array(target_full, dtype=wp.float32, device=self.model.device))

    def _collect_collider_impulses(self):
        impulses, pos, ids = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
        self.collider_impulse_ids.fill_(-1)
        n = min(impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n].assign(impulses[:n])
        self.collider_impulse_pos[:n].assign(pos[:n])
        self.collider_impulse_ids[:n].assign(ids[:n])

    def simulate_robot(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            self.body_sand_forces.assign(self.state_0.body_f)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        wp.launch(
            subtract_body_force,
            dim=self.sand_state_0.body_q.shape,
            inputs=[
                self.frame_dt,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.body_sand_forces,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                self.sand_state_0.body_q,
                self.sand_state_0.body_qd,
            ],
        )
        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)
        self._collect_collider_impulses()

    def step(self):
        self.apply_control()
        self.simulate_robot()
        self.simulate_sand()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _log_row(self):
        if self.log_writer is None:
            return
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        base_q = body_q[self.center_link_body]
        base_qd = body_qd[self.center_link_body]
        joint_q_np = self.state_0.joint_q.numpy()
        joint_pos = [float(joint_q_np[self.joint_q_indices[name]]) for name in JOINT_NAMES]

        sand_force = np.zeros(3, dtype=np.float64)
        sand_torque = np.zeros(3, dtype=np.float64)
        body_f_np = self.body_sand_forces.numpy()
        for b in self.leg_body_indices:
            sand_force += body_f_np[b][0:3]
            sand_torque += body_f_np[b][3:6]

        row = [self.frame_index, self.sim_time]
        row += list(base_q[0:3])
        row += list(base_q[3:7])
        row += list(base_qd[0:3])
        row += list(base_qd[3:6])
        row += joint_pos
        row += list(self._last_targets)
        row += list(sand_force)
        row += list(sand_torque)
        self.log_writer.writerow(row)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()
        self._log_row()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--hexapod-usd", type=str, default=HEXAPOD_USD_PATH)
        parser.add_argument("--gait-csv", type=str, default=DEFAULT_GAIT_CSV)
        parser.add_argument("--gait-dt", type=float, default=0.01)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.015)
        parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
        parser.add_argument("--grid-type", "-gt", choices=["sparse", "dense", "fixed"], default="fixed")
        parser.add_argument("--max-active-cells", type=int, default=1 << 16)
        parser.add_argument("--max-collider-nodes", type=int, default=1 << 18)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument("--sand-density", type=float, default=1600.0)
        parser.add_argument("--sand-lo", type=float, nargs=3, default=[-0.3, -0.3, 0.0], metavar=("X", "Y", "Z"))
        parser.add_argument("--sand-hi", type=float, nargs=3, default=[1.2, 0.3, 0.08], metavar=("X", "Y", "Z"))
        parser.add_argument("--log-csv", type=str, default=None, help="Path to write per-frame dynamics CSV log.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        if example.log_file is not None:
            example.log_file.close()
