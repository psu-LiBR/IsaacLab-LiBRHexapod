# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Blender scripting-tab helper: paste into Blender's scripting editor and Run.

Prints a JSON of per-body world-rotation matrices that render_pyvista.py uses
as the local-frame correction (--local-fix).  Save the printed JSON to
hexapod-assets/OBJ/local_fix.json.
"""

import json

import bpy

BODIES = [
    "CenterLink",
    "BackLink",
    "FrontLink",
    "MiddleLeft",
    "MiddleRight",
    "BackLeft",
    "BackRight",
    "FrontLeft",
    "FrontRight",
]

result = {}
for name in BODIES:
    obj = bpy.data.objects.get(name)
    if obj is None:
        print(f"MISSING: {name}")
        continue
    rot = obj.matrix_world.to_3x3().normalized()
    result[name] = [list(row) for row in rot]

print(json.dumps(result, indent=2))
