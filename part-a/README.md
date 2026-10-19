# Part A: human motion to robot

A motion for the Unitree G1 was generated in UltimateBots Studio, exported as a LAFAN CSV, and played
back in MuJoCo with the G1 model from Unitree RL Gym.

| File | What |
|---|---|
| [`motion.mp4`](motion.mp4) | rendered result |
| [`motion.csv`](motion.csv) | exported LAFAN motion, <!-- TODO: N frames, N seconds --> |
| [`prompt.txt`](prompt.txt) | <!-- TODO: the text prompt, or a description of the source video --> |
| [`play_motion.py`](play_motion.py) | the playback and rendering script |

## Reproduce

Setup is in the [top-level README](../README.md). Then:

```bash
export MUJOCO_GL=cgl      # macOS; use egl on headless Linux
python part-a/play_motion.py part-a/motion.csv --out part-a/motion.mp4
```

The script prints a summary of every column group before it renders, and writes the video at the
CSV's 30 fps.

## How it works

1. **Load the robot.** `unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml`, the 29-joint G1
   (36 position coordinates: 7 for the free-floating base plus 29 joints).
2. **Read the CSV.** One row per frame.
3. **Set the pose.** For each frame, write the row straight into MuJoCo's `qpos` (base position, base
   orientation, joint angles) and call `mj_forward` to update the geometry. This is *kinematic*
   playback. There is no physics and no controller, so the robot shows exactly what the file says,
   including any foot sliding or floating that the retargeting left in.
4. **Render.** Offscreen camera that follows the robot, frames written to MP4.

## How did you determine what the columns in the CSV represent?

<!-- TODO: after exporting, confirm each point against the real file's output of play_motion.py and edit. -->

I did not assume a layout. I read the file, then checked each hypothesis against the robot model:

1. **Shape.** Every row has 36 numbers. That equals the G1 29-DoF model's `nq`: 3 (base position)
   + 4 (base quaternion) + 29 (joints). No header, no time column.
2. **Columns 0–2 are the base position.** Values stay around 0.7–0.8 in the third column (the height of
   the pelvis above the floor in metres, matching the `0.793` in the XML) while the first two change as
   the robot moves.
3. **Columns 3–6 are a unit quaternion.** Their norm is 1.000 on every frame, which no other group of
   four columns would have. Which one is `w` is decided by the data. Standing upright, the three small
   values are the vector part and the one close to 1 is `w`. That is the **last** column, so the order is
   `x, y, z, w`. MuJoCo expects `w, x, y, z`, so the script reorders it.
4. **Columns 7–35 are the 29 joint angles in radians, in the order of the XML.** That order is the six
   left-leg joints, six right-leg, three waist, seven left-arm and seven right-arm. Evidence: every
   column stays inside the joint limit written in the XML for its position (the script flags any that
   do not), and the ranges are physically plausible per joint. Knee columns only go positive (the knee
   bends one way), and the arm columns move only when the motion uses the arms. Playing it also
   confirms it, because a wrong order would twist the robot into a pose no human makes.
5. **Frame rate.** LAFAN data is 30 fps. Played at 30 fps the speed of the motion looks natural, and
   the root displacement per frame is consistent with walking speed.

<!-- TODO: paste the summary printed by play_motion.py for the real file here. -->
