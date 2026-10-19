# Part B: walking in a circle

**Video:** [`circle_walk.mp4`](circle_walk.mp4) (30 s). The yellow dots are the path of the robot's base.
**Baseline:** [`forward.mp4`](forward.mp4), the default policy walking straight.
**Path plot:** [`circle_walk.png`](circle_walk.png).

## What I changed, and why it gives a circle

One line in the deployment config. In [`g1_forward.yaml`](g1_forward.yaml) (a copy of upstream `g1.yaml`) the velocity command is

```yaml
cmd_init: [0.5, 0, 0]      # vx, vy, wz : forward only
```

and in [`g1_circle.yaml`](g1_circle.yaml) it is

```yaml
cmd_init: [0.5, 0, 0.5]    # forward 0.5 m/s and turn left 0.5 rad/s
```

The policy was trained to follow a command `[vx, vy, wz]`: forward speed (m/s), sideways speed (m/s)
and yaw rate (rad/s). It is not told where to go, only how to move. If the robot moves forward at a
constant speed `v` while turning at a constant rate `ω`, its path is a circle of radius `R = v / ω`.
With `v = 0.5` and `ω = 0.5` the target radius is 1 m. The weights were not touched and no new policy
was trained.

The upstream script `deploy_mujoco.py` opens a live viewer and imports `legged_gym`. I wrote
[`run_policy.py`](run_policy.py), which has the same observation, policy and PD loop but renders
offscreen to MP4 and does not need the training package.

**Check.** I fitted a circle to the base path (30 s, upright at the end):

| Run | Commanded | Measured speed | Measured yaw rate | Fitted radius |
|---|---|---|---|---|
| forward | `[0.5, 0, 0]` | 0.48 m/s | 0.01 rad/s | 35.9 m (almost straight) |
| circle | `[0.5, 0, 0.5]` | 0.47 m/s | 0.44 rad/s | **1.08 m** |

The radius is 1.08 m rather than 1.00 m because the policy tracks the command only approximately:
it walks slightly slower and turns slightly slower than asked, and `0.47 / 0.44 ≈ 1.08`. The small
ripple on the path is the left–right sway of each step.

## What inputs does the policy receive, and where do they come from?

An observation vector of 47 numbers, built every control step (50 Hz) in `run_policy.py`
(upstream: `deploy_mujoco.py`):

| Slice | Size | Content | Source |
|---|---|---|---|
| 0–2 | 3 | base angular velocity (× 0.25) | simulator state `qvel[3:6]`; on the real robot, the IMU gyro |
| 3–5 | 3 | gravity direction in the body frame | computed from the base orientation quaternion `qpos[3:7]`; tells the policy how it is tilted |
| 6–8 | 3 | velocity command `[vx, vy, wz]` × `[2, 2, 0.25]` | **set by me** in `cmd_init` (on the real robot: a joystick) |
| 9–20 | 12 | joint positions minus default pose | simulator `qpos[7:]`; real robot: joint encoders |
| 21–32 | 12 | joint velocities (× 0.05) | simulator `qvel[6:]`; real robot: joint encoders |
| 33–44 | 12 | previous action | stored from the last policy call |
| 45–46 | 2 | gait phase `sin`, `cos` of `2π·(t mod 0.8 s)/0.8 s` | the clock; gives the policy a rhythm to step to |

Everything except the command comes from the robot's own sensing. The policy is not given its
position or its linear velocity, so it cannot know where the circle is. It only sees the command.

## What does the policy output, and how does it become joint torque?

The policy (a TorchScript network) outputs **12 numbers**, one per leg joint. They are not torques.
They are scaled offsets from the standing pose:

```
target_angle = action * 0.25 + default_angle          (action_scale = 0.25)
```

A PD controller then turns the target into a torque for each joint. It runs every simulation step
(500 Hz), while the policy updates every 10th step (50 Hz):

```
torque = kp * (target_angle - joint_angle) - kd * joint_velocity
```

`kp` and `kd` are per-joint gains in the config, for example `kp = 100, kd = 2` for the hip joints
and `kp = 150, kd = 4` for the knees. The torque is written to `data.ctrl`, which MuJoCo applies to
the actuators.

## Reproduce

```bash
python part-b/run_policy.py part-b/g1_forward.yaml --out part-b/forward.mp4
python part-b/run_policy.py part-b/g1_circle.yaml  --out part-b/circle_walk.mp4
```

(Setup is in the top-level [README](../README.md).)
