# Bathsheba Project RoboAI

A robotics project focused on 7-DOF manipulator control, planning, and perception for compliant robotic manipulation. The repository includes custom kinematic and trajectory tooling, impedance controllers, environment setups, and perception utilities for robot interaction tasks.

## Overview

- Custom kinematic stack for 7-DOF manipulators
- Verified inverse kinematics (IK) solvers
- Impedance control architectures for compliant manipulation
- Perception pipeline for pose estimation and camera-based object detection
- Simulation and trajectory planning support for robotic tasks

## Repository structure

```text
Bathsheba-Project-roboai/
├── README.md
├── requirements.txt
├── assets/
│   ├── menagerie/
│   ├── mujoco_api_reference.txt
│   └── scene_pick.xml
├── control/
│   ├── __init__.py
│   ├── CS_impedance_controller.py
│   ├── grasp_check.py
│   └── JS_impedance_controller.py
├── envs/
│   ├── __init__.py
│   └── reach_env.py
├── perception/
│   ├── __init__.py
│   ├── camera_stream.py
│   ├── detector.py
│   └── pose_estimator.py
├── planning/
│   ├── __init__.py
│   ├── cartesian_trajectory.py
│   ├── fk.py
│   ├── ik.py
│   └── trajectory.py
├── scripts/
│   ├── run.py
│   └── test_control_planning/
│       ├── test_cartesian_trajectory.py
│       ├── test_circle.py
│       ├── test_trajectory.py
│       ├── test1_setpointtracking.py
│       ├── test2_dampingandsweep.py
│       ├── test3_orientationtracking.py
│       └── test4_JSImpedance.py
├── test_perception/
│   └── test_pose_estimator.py
└── utils/
    ├── __init__.py
    └── transform.py
```

## Folder breakdown

### control/
Contains robot control logic, including compliant impedance controllers and grasp checks.

### envs/
Defines simulation and task environments used for robot interaction experiments.

### perception/
Includes camera streaming, object detection, and pose estimation utilities.

### planning/
Contains forward and inverse kinematics, trajectory generation, and Cartesian planning logic.

### scripts/
Houses runnable scripts and test cases for control, trajectory planning, and evaluation.

### utils/
General helper functions, such as transforms and geometric utilities.

### assets/
Static assets, scene descriptions, and supporting reference files for simulation or experiments.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- Initial validation was performed on a Franka Emika Panda 7-DOF robotic arm.
- The repository is actively evolving, especially around perception and vision stack integration.
- Contributions are welcome.

