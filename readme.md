# A demo of controlling a robot in Mujoco using a predefined policy.

# Requirements

```bash
conda env create -f environment.yml
```

# Usage

```bash
# Run the demo
python main.py
```

# Notes
- How to control the robot:
  - You can use a joystick or keyboard to control the robot.
- The robot will follow a predefined policy(`policy_dh.jit`).
- The robot will be controlled in Mujoco using the `mujoco` package.