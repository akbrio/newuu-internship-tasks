# NewUU Robotics Internship: Junior Track

Unitree G1 humanoid in MuJoCo.

| Part | What | Where |
|---|---|---|
| A | Generated motion played on the G1 and rendered to video | [`part-a/`](part-a/README.md) |
| B | Pretrained walking policy driven to walk in a circle | [`part-b/`](part-b/NOTE.md) |

## Repository layout

```
part-a/
  motion.mp4        rendered result
  motion.csv        LAFAN motion exported from UltimateBots Studio
  prompt.txt        text prompt (or description of the source video) used
  play_motion.py    loads the G1 in MuJoCo, plays the CSV, renders the video
  README.md         how to reproduce, and how the CSV columns were identified
part-b/
  circle_walk.mp4   G1 walking in a circle   (forward.mp4 = default behaviour)
  g1_circle.yaml    the changed config      (g1_forward.yaml = upstream default)
  run_policy.py     headless MuJoCo runner for the pretrained policy
  NOTE.md           what changed, why it works, policy inputs and outputs
```

## Setup (macOS or Linux, Python 3.11)

```bash
git clone https://github.com/<your-user>/<this-repo>.git && cd <this-repo>

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Robot model and pretrained policy come from Unitree's repository.
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git

# macOS only: use the native OpenGL backend for offscreen rendering.
export MUJOCO_GL=cgl            # on Linux without a display: export MUJOCO_GL=egl
```

Nothing in this repository modifies `unitree_rl_gym`; the scripts read the robot model
(`resources/robots/g1_description/`) and the policy (`deploy/pre_train/g1/motion.pt`) from it.

## Reproduce

```bash
python part-a/play_motion.py part-a/motion.csv --out part-a/motion.mp4
python part-b/run_policy.py part-b/g1_circle.yaml --out part-b/circle_walk.mp4
```

Tested with MuJoCo 3.15.0 and PyTorch 2.14.1 on an Apple M2 Pro. Rendering 30 s of policy
rollout takes about 20 s.

## Credits

Robot model and pretrained policy: [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
(BSD-3-Clause). The Part B runner follows its `deploy_mujoco.py`. Motion generation:
[UltimateBots Studio](https://studio.ultimatebots.com/home).
