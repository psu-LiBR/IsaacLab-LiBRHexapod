# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline PyVista renderer for hexapod body-pose logs.

Reads an .npz produced by playpyvista.py (per-frame world pose of every
articulation body) and animates the per-link OBJ meshes in hexapod-assets/OBJ
to produce an .mp4 or a PNG frame sequence. Rendering goes through VTK's
standard OpenGL path, not Isaac Sim's RTX pipeline, so it works around the
rtx_scenedb_plugin driver crash.

This script has no Isaac Sim dependency -- run it with plain Python, not
isaaclab.bat:

    python scripts/reinforcement_learning/rsl_rl/render_pyvista.py \
        --npz logs/rsl_rl/hexapod_mimic/<run>/body_pose_log/body_poses_<ts>.npz \
        --local-fix hexapod-assets/OBJ/local_fix.json \
        --out hexapod_render.mp4

Requires: pip install pyvista numpy imageio imageio-ffmpeg

Per-link orientation fix (--local-fix):
    OBJ files are exported from Blender in each link's own local/object-space
    coordinate frame.  Isaac Sim's body_quat_w describes each link's orientation
    in world space using the USD body frame, which differs from the OBJ export
    frame for every link that isn't the root.  Without the fix, applying
    body_quat_w directly to OBJ vertices produces wrong per-link orientations
    (positions are correct, orientations are not).

    The fix: for each link i, compute
        R_fix[i] = R(body_quat_w[0, i]).T  @  R_blender[i]
    where R_blender[i] is the link's world-rotation matrix from Blender's
    matrix_world at the assembled default pose.  Pre-applying R_fix.T to the
    rest vertices means apply_frame() needs no change: it still does
        v_world = rest_points @ R(quat_w[t]).T + pos_w[t]
    which now expands to
        v_world = v_local @ R_fix.T @ R(quat_w[t]).T + pos_w[t]
               = R(quat_w[t]) @ R_fix @ v_local  +  pos_w[t]  (column-vector equiv.)

    To generate local_fix.json, open the Blender file that contains the
    assembled hexapod, switch to the Scripting workspace, and run
        hexapod-assets/OBJ/extract_blender_rotations.py
    Copy the printed JSON to hexapod-assets/OBJ/local_fix.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    parser = argparse.ArgumentParser(description="Render a hexapod body-pose log with PyVista.")
    parser.add_argument("--npz", type=str, required=True, help="Path to body_poses_*.npz from playpyvista.py.")
    parser.add_argument(
        "--obj_dir", type=str, default="hexapod-assets/OBJ", help="Directory of per-link OBJ meshes."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output video path (.mp4), or output directory if --frames is set. Defaults to"
            " 'render.mp4' (or 'render_frames/' with --frames) saved next to --npz."
        ),
    )
    parser.add_argument("--frames", action="store_true", help="Write a PNG sequence to --out instead of an .mp4.")
    parser.add_argument("--fps", type=float, default=None, help="Override frame rate (default: 1/dt from the log).")
    parser.add_argument("--color", type=str, default="lightgray", help="Mesh color.")
    parser.add_argument("--window_size", type=int, nargs=2, default=(1280, 720), help="Render resolution (w h).")
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Multiply the auto-fitted camera distance by this factor. >1 zooms out, <1 zooms in. Default 1.0.",
    )
    parser.add_argument("--ground", action="store_true", help="Render a flat ground plane at z=0.")
    parser.add_argument(
        "--ground-size", type=float, default=5.0, metavar="M", help="Side length of the ground plane in meters."
    )
    parser.add_argument("--ground-color", type=str, default="tan", help="Ground plane color.")
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.25,
        metavar="M",
        help="Grid line spacing on the ground plane in meters (default 0.25).",
    )
    parser.add_argument("--grid-color", type=str, default="gray", help="Ground grid line color.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Render frame 0 from four camera angles (iso / top / front / side) as a single "
            "PNG montage saved next to --npz. Use this to diagnose axis/orientation issues "
            "before committing to a full video render."
        ),
    )
    parser.add_argument(
        "--local-fix",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Path to local_fix.json produced by extract_blender_rotations.py. "
            "Provides per-body Blender world-rotation matrices that correct the "
            "mismatch between OBJ local axes and USD body-frame axes. "
            "Without this, link orientations will be wrong for all non-root links."
        ),
    )
    parser.add_argument(
        "--obj-up-axis",
        choices=["Y", "Z"],
        default="Y",
        help=(
            "Up axis used when the OBJ files were exported from Blender. "
            "Y (default): the legacy / Blender-5 default where OBJ +Y = Blender local +Z "
            "and OBJ +Z = Blender local -Y. "
            "Z: no axis conversion (OBJ axes match Blender local axes). "
            "Use Z only if you explicitly set Forward=Y, Up=Z in the Blender OBJ exporter."
        ),
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz).resolve().parent
    if args.out is None:
        args.out = str(npz_dir / ("render_frames" if args.frames else "render.mp4"))

    data = np.load(args.npz)
    body_pos_w = data["body_pos_w"]  # (T, num_bodies, 3)
    body_quat_w = data["body_quat_w"]  # (T, num_bodies, 4) wxyz
    body_names = [str(n) for n in data["body_names"]]
    dt = float(data["dt"])
    fps = args.fps if args.fps is not None else 1.0 / dt

    obj_dir = Path(args.obj_dir)
    meshes: dict[str, pv.PolyData] = {}
    rest_points: dict[str, np.ndarray] = {}
    for name in body_names:
        obj_path = obj_dir / f"{name}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"Missing OBJ for body '{name}': {obj_path}")
        mesh = pv.read(str(obj_path))
        meshes[name] = mesh
        rest_points[name] = mesh.points.copy()

    # --obj-up-axis Y (default): Blender's legacy OBJ exporter stores vertices with
    # OBJ +Y = Blender local +Z (up) and OBJ +Z = Blender local -Y (depth).
    # Pre-multiply rest_points by M.T so subsequent transforms work in Blender-local
    # coordinates:  v_blender = M @ v_obj  where M = [[1,0,0],[0,0,-1],[0,1,0]]
    if args.obj_up_axis == "Y":
        M_to_blender = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [0.0, 1.0,  0.0]])
        for name in body_names:
            rest_points[name] = rest_points[name] @ M_to_blender.T

    # --local-fix: per-body correction for OBJ-local vs USD-body-frame mismatch.
    # For each link i:  R_fix[i] = R(quat_w[0,i]).T @ R_blender[i]
    # Pre-multiply rest_points by R_fix[i].T so apply_frame is unchanged:
    #   v_world = rest_points_fixed @ rot_t.T + pos
    #           = v_local @ R_fix.T @ rot_t.T + pos
    #           ≡ R(quat_w[t]) @ R_fix @ v_local + pos   (column-vector equiv.)
    if args.local_fix is not None:
        with open(args.local_fix) as fh:
            blender_rots: dict = json.load(fh)
        for i, name in enumerate(body_names):
            if name not in blender_rots:
                raise ValueError(
                    f"--local-fix JSON is missing an entry for body '{name}'. "
                    f"Re-run extract_blender_rotations.py and make sure all 9 bodies are present."
                )
            R_blender = np.array(blender_rots[name], dtype=np.float64)  # 3×3 from Blender matrix_world
            R_quat0 = quat_to_rotmat(body_quat_w[0, i])
            R_fix = R_quat0.T @ R_blender
            rest_points[name] = rest_points[name] @ R_fix.T
        print(f"[INFO] Per-body orientation fix applied from: {args.local_fix}")

    def apply_frame(t: int) -> None:
        for i, name in enumerate(body_names):
            rot_t = quat_to_rotmat(body_quat_w[t, i])
            meshes[name].points = rest_points[name] @ rot_t.T + body_pos_w[t, i]

    # Ground plane mesh: static, placed once at the robot's frame-0 XY center so
    # the robot visibly travels across it.  Make --ground-size large enough to
    # cover the full travel distance (robot walks ~0.2 m/s; default 5 m covers
    # ~25 s at that speed).
    ground_mesh: pv.PolyData | None = None
    if args.ground:
        start_xy = body_pos_w[0].mean(axis=0)
        n_cells = max(1, round(args.ground_size / args.grid_spacing))
        ground_mesh = pv.Plane(
            center=(float(start_xy[0]), float(start_xy[1]), 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=args.ground_size,
            j_size=args.ground_size,
            i_resolution=n_cells,
            j_resolution=n_cells,
        )

    # --diagnose: render frame 0 from four angles as a single montage PNG.
    # Use this to determine whether link orientations are correct before
    # committing to a full video render.
    if args.diagnose:
        apply_frame(0)
        tiles = []
        VIEWS = {
            "iso":   lambda pl: pl.view_isometric(),
            "top":   lambda pl: pl.view_xy(),
            "front": lambda pl: pl.view_xz(),
            "side":  lambda pl: pl.view_yz(),
        }
        tw, th = args.window_size[0] // 2, args.window_size[1] // 2
        for label, set_view in VIEWS.items():
            pl = pv.Plotter(off_screen=True, window_size=(tw, th))
            for name in body_names:
                pl.add_mesh(meshes[name].copy(), color=args.color, smooth_shading=True)
            if ground_mesh is not None:
                pl.add_mesh(
                    ground_mesh.copy(),
                    color=args.ground_color,
                    show_edges=True,
                    edge_color=args.grid_color,
                    line_width=1.0,
                )
            pl.reset_camera()
            set_view(pl)
            pl.add_text(label, font_size=10, color="black")
            tiles.append(pl.screenshot(return_img=True))
            pl.close()
        top_row = np.concatenate([tiles[0], tiles[1]], axis=1)
        bot_row = np.concatenate([tiles[2], tiles[3]], axis=1)
        montage = np.concatenate([top_row, bot_row], axis=0)
        diag_path = str(Path(args.npz).resolve().parent / "diagnose_frame0.png")
        import imageio.v2 as iio
        iio.imwrite(diag_path, montage)
        print(f"[INFO] Diagnostic montage saved: {diag_path}")
        return

    plotter = pv.Plotter(off_screen=True, window_size=tuple(args.window_size))
    for name in body_names:
        plotter.add_mesh(meshes[name], color=args.color, smooth_shading=True)
    if ground_mesh is not None:
        plotter.add_mesh(
            ground_mesh,
            color=args.ground_color,
            show_edges=True,
            edge_color=args.grid_color,
            line_width=1.0,
        )

    # Apply frame 0 BEFORE framing the camera -- otherwise reset_camera() fits
    # to the raw, untransformed local-frame meshes (a tiny cluster near each
    # link's own origin) instead of the actual assembled robot.
    apply_frame(0)
    plotter.reset_camera()
    plotter.view_isometric()

    # Fixed camera offset re-applied around a moving focal point each frame.
    # --camera-distance scales the auto-fitted distance so the robot appears
    # smaller/larger in frame without changing the viewing angle.
    cam_offset = np.array(plotter.camera.position) - np.array(plotter.camera.focal_point)
    cam_offset *= args.camera_distance

    num_frames = body_pos_w.shape[0]
    if args.frames:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        plotter.open_movie(args.out, framerate=fps)

    for t in range(num_frames):
        apply_frame(t)
        center = body_pos_w[t].mean(axis=0)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.position = tuple(center + cam_offset)

        if args.frames:
            plotter.screenshot(str(out_dir / f"frame_{t:05d}.png"))
        else:
            plotter.write_frame()

    if not args.frames:
        plotter.close()

    print(f"[INFO] Rendered {num_frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
