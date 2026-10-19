"""Play a LAFAN-format motion CSV on the Unitree G1 in MuJoCo and render it to MP4.

The CSV has one row per frame. For the G1 (29 DoF) a row is 36 numbers:

    [0:3]   root position x, y, z (metres)
    [3:7]   root orientation quaternion (x, y, z, w)
    [7:36]  29 joint angles (radians), in the joint order of g1_29dof.xml

The script first prints a summary of every column group, which is also how the
column meaning was checked (see README). The motion is played kinematically:
each frame is written straight into qpos, so no physics or controller is used.

Usage:
    python part-a/play_motion.py part-a/motion.csv --out part-a/motion.mp4
"""
import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL_GYM = REPO_ROOT / "unitree_rl_gym"
MODEL_XML = "resources/robots/g1_description/g1_29dof.xml"
N_JOINTS = 29
N_COLS = 7 + N_JOINTS


def load_csv(path):
    """Read the CSV as a float array, skipping a header row if there is one."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                rows.append([float(v) for v in row if v.strip() != ""])
            except ValueError:
                continue  # header or comment line
    data = np.array(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != N_COLS:
        raise SystemExit(f"expected {N_COLS} columns per row, got shape {data.shape}")
    return data


def build_model(rl_gym):
    """G1 29-DoF model from Unitree RL Gym (it already contains a floor and light)."""
    spec = mujoco.MjSpec.from_file(str(rl_gym / MODEL_XML))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    return spec.compile()


def summarise(data, model, fps):
    """Print the evidence used to decide what each column group means."""
    names = [model.joint(i).name for i in range(1, model.njnt)]
    print(f"{len(data)} frames x {data.shape[1]} columns -> {len(data) / fps:.1f} s at {fps} fps")
    print("root xyz  min:", data[:, 0:3].min(0).round(3), " max:", data[:, 0:3].max(0).round(3))
    qnorm = np.linalg.norm(data[:, 3:7], axis=1)
    print(f"cols 3-6 quaternion norm: min {qnorm.min():.4f} max {qnorm.max():.4f} (1.0 means unit quaternion)")
    print(f"cols 3-6 mean |value| per column (x,y,z,w): {np.abs(data[:, 3:7]).mean(0).round(3)}")
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    q = data[:, 7:]
    outside = ((q < lo - 1e-3) | (q > hi + 1e-3)).sum(0)
    print("joint columns outside the XML joint limits (count of frames):")
    for i, n in enumerate(names):
        flag = "" if outside[i] == 0 else "   <-- check"
        print(f"  col {7 + i:2d} {n:28s} range [{q[:, i].min():6.2f}, {q[:, i].max():6.2f}]"
              f"  limit [{lo[i]:6.2f}, {hi[i]:6.2f}]{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output .mp4 path")
    ap.add_argument("--fps", type=int, default=30, help="frame rate of the CSV (LAFAN uses 30)")
    ap.add_argument("--rl-gym", type=Path, default=DEFAULT_RL_GYM, help="path to the unitree_rl_gym clone")
    ap.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw")
    args = ap.parse_args()

    data = load_csv(args.csv)
    model = build_model(args.rl_gym)
    summarise(data, model, args.fps)
    mj_data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.5, 150, -15

    frames = []
    for row in data:
        mj_data.qpos[0:3] = row[0:3]
        if args.quat_order == "xyzw":  # MuJoCo wants w, x, y, z
            mj_data.qpos[3:7] = [row[6], row[3], row[4], row[5]]
        else:
            mj_data.qpos[3:7] = row[3:7]
        mj_data.qpos[7:] = row[7:]
        mujoco.mj_forward(model, mj_data)
        cam.lookat[:] = [row[0], row[1], 0.8]
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
