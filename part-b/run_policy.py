"""Run the pretrained Unitree G1 walking policy in MuJoCo and render it to MP4.

This is a headless version of unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py.
The control loop (observation, policy, PD torque) is the same as upstream; the
differences are that it does not import legged_gym, it renders offscreen to a
video instead of opening a viewer, and it draws the path the robot walked.

Usage:
    python part-b/run_policy.py part-b/g1_forward.yaml --out part-b/forward.mp4
    python part-b/run_policy.py part-b/g1_circle.yaml  --out part-b/circle_walk.mp4
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL_GYM = REPO_ROOT / "unitree_rl_gym"
GAIT_PERIOD = 0.8  # seconds, same constant as upstream
VIDEO_FPS = 25


def gravity_in_body_frame(quat):
    """Gravity direction expressed in the robot's base frame (quat = w, x, y, z)."""
    qw, qx, qy, qz = quat
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def pd_torque(target_q, q, kp, dq, kd):
    """Position target -> torque. Target joint velocity is zero."""
    return (target_q - q) * kp - dq * kd


def load_config(path, rl_gym):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("policy_path", "xml_path"):
        cfg[key] = cfg[key].replace("{LEGGED_GYM_ROOT_DIR}", str(rl_gym))
    return cfg


def add_trail(scene, points, rgba=(1.0, 0.8, 0.1, 1.0), radius=0.015):
    """Draw the walked path as small spheres on the floor."""
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]), np.array([p[0], p[1], 0.02]),
            np.eye(3).flatten(), np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--rl-gym", type=Path, default=DEFAULT_RL_GYM, help="path to the unitree_rl_gym clone")
    ap.add_argument("--duration", type=float, default=30.0, help="simulated seconds")
    ap.add_argument("--out", type=Path, required=True, help="output .mp4 path")
    ap.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "WZ"), help="override cmd_init")
    ap.add_argument("--camera", choices=["overview", "follow"], default="overview")
    args = ap.parse_args()

    cfg = load_config(args.config, args.rl_gym)
    dt, decimation = cfg["simulation_dt"], cfg["control_decimation"]
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    cmd = np.array(args.cmd if args.cmd else cfg["cmd_init"], dtype=np.float32)
    n_act, n_obs = cfg["num_actions"], cfg["num_obs"]
    print(f"command [vx, vy, wz] = {cmd.tolist()}")

    model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    policy = torch.jit.load(cfg["policy_path"])

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 90, -40
    if args.camera == "overview":
        # circle centre is at (0, vx/wz) for a robot starting at the origin facing +x
        radius = abs(cmd[0] / cmd[2]) if abs(cmd[2]) > 1e-6 else 0.0
        cam.lookat[:] = [0.0, np.sign(cmd[2]) * radius, 0.6]
        cam.distance = max(4.0, 2.2 * radius + 3.0)
    else:
        cam.distance = 3.0

    action = np.zeros(n_act, dtype=np.float32)
    target_q = default_angles.copy()
    obs = np.zeros(n_obs, dtype=np.float32)
    n_steps = int(args.duration / dt)
    frame_every = int(round(1.0 / VIDEO_FPS / dt))
    path, frames = [], []

    for step in range(1, n_steps + 1):
        data.ctrl[:] = pd_torque(target_q, data.qpos[7:], kps, data.qvel[6:], kds)
        mujoco.mj_step(model, data)

        if step % decimation == 0:
            phase = (step * dt) % GAIT_PERIOD / GAIT_PERIOD
            obs[0:3] = data.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_in_body_frame(data.qpos[3:7])
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + n_act] = (data.qpos[7:] - default_angles) * cfg["dof_pos_scale"]
            obs[9 + n_act:9 + 2 * n_act] = data.qvel[6:] * cfg["dof_vel_scale"]
            obs[9 + 2 * n_act:9 + 3 * n_act] = action
            obs[9 + 3 * n_act:] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target_q = action * cfg["action_scale"] + default_angles

        if step % frame_every == 0:
            path.append(data.qpos[:3].copy())
            if args.camera == "follow":
                cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            add_trail(renderer.scene, path[::2])
            frames.append(renderer.render())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=VIDEO_FPS, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)")

    path = np.array(path)
    fell = path[-1, 2] < 0.5  # pelvis height; standing is ~0.79 m
    print(f"final base height {path[-1, 2]:.2f} m -> {'FELL' if fell else 'upright'}")
    summarise_path(path, args.out.with_suffix(".png"))


def summarise_path(path, png):
    """Fit a circle to the xy path (algebraic least squares) and plot it."""
    xy = path[:, :2]
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r_fit = np.sqrt(c + cx ** 2 + cy ** 2)
    resid = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r_fit)
    duration = (len(xy) - 1) / VIDEO_FPS
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / duration
    print(f"measured mean speed {speed:.2f} m/s, mean yaw rate {speed / r_fit:.2f} rad/s (speed / radius)")
    print(f"circle fit: centre=({cx:.2f}, {cy:.2f}) radius={r_fit:.2f} m, max deviation {resid.max():.2f} m")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xy[:, 0], xy[:, 1], lw=2, label="robot base path")
    ax.plot(*xy[0], "go", label="start")
    ax.add_patch(plt.Circle((cx, cy), r_fit, fill=False, ls="--", color="gray", label=f"fit, r = {r_fit:.2f} m"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=150)


if __name__ == "__main__":
    main()
