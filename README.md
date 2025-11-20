# LiDAR Team - TritonAI

## Project Structure

```
└── lidar-team/
    ├── ros2_ws                     # ROS 2 packages and scripts
        └── lidar_ml_bridge         # Subscriber Node for LiDAR -> Vectorized Points
    └── .git/         # Git version history
```

## Contributing

Checklist for contributors
- Fork or clone the repository and create a feature branch per change.
- Keep `ros2_ws/src/` source files under version control; do not commit `build/`, `install/`, or `log/`.
- Add tests for parsing/logic where feasible (see `lidar_ml_bridge/test`).

Environment setup (recommended)
1. Install ROS 2 matching the project target (e.g., Humble/Rolling/etc.). Follow official ROS 2 docs for your OS.
2. Create a Python virtual environment for isolation (optional but recommended):

```bash
python3 -m venv ~/.venv/lidar-team
source ~/.venv/lidar-team/bin/activate
pip install -U pip setuptools
```

Workspace setup
1. From the repo root:

```bash
cd ros2_ws
rosdep update || true
rosdep install --from-paths src --ignore-src -r -y
```

2. Build the workspace:

```bash
colcon build --symlink-install
source install/setup.bash
```

Run the LiDAR subscriber node

```bash
# (in a shell where `source install/setup.bash` was run)
ros2 run lidar_ml_bridge lidar_pointcloud_bridge --ros-args -p pointcloud_topic:=/lidar/points
```

Testing
- Unit tests are under `ros2_ws/src/lidar_ml_bridge/test`. Run them with pytest from the package directory or using colcon:

```bash
cd ros2_ws
colcon test --packages-select lidar_ml_bridge
colcon test-result --verbose
```

Code style and linting
- Follow existing project style. We run basic flake8/PEP8 checks in CI. Run locally with `pytest` & `flake8` where configured.

Commit & PR guidelines
- Use small, focused commits and clear messages.
- Open a PR targeting `main`. Include a short description, the motivation, and which files changed.
- If the change affects runtime behavior (topics, parameters, message definitions), add a brief "Runtime impact" note in the PR description.
- Ensure tests pass and add tests for new logic where reasonable.

Large datasets and secrets
- Do NOT commit recorded bags, datasets, or secrets. The repo .gitignore excludes common large artifacts (bag files, pcap, pretrained models).

Help & debugging
- If you hit build issues, include the output of `colcon build` and `ros2 doctor` in the PR or an issue.


## Team Credits

**Team Lead:** Benjamin Crawford

**Members:** Evan Chou, Michael Lai

*Funded and developed within Triton AI*