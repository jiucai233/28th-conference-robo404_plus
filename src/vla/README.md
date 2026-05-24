# Vision-Language-Action (VLA) ROS 2 Controller with Apple MLX

A production-ready ROS 2 Humble package implementing a Vision-Language-Action (VLA) classification model using Apple MLX, decoupled business logic, and a safety arbitration layer for obstacle avoidance and control boundaries.

## Why do it?

in reality, there are a lot of scenario that requires the robot to understand the semantic meaning of the visual input, like a plastic bag on the road, it will make the traditional line following robot to stop, but the VLA model can understand the semantic meaning of the visual input and make the decision to ignore the plastic bag and continue to follow the line. Besides, a person at the edge of the frame but playing the phone while walking, yolo can only recognize it as a single human but vla model can make the robot stop before that person to prevent potential accidents. Also the model can applies to the scenario that includes a traffic police directing.

## Discrete Action Space (VLAAction)

The VLA model outputs one of 4 discrete semantic action states as tokens:

1. `<STATE_FOLLOW_LINE>`: Default state. Trust local PID line tracking.
2. `<STATE_STOP_LIGHT>`: Semantic stop for traffic lights/signs.
3. `<STATE_STOP_OBSTACLE>`: Stop for physical/dynamic obstacles.
4. `<STATE_IGNORE>`: Ignore visual false-positives (shadows, flat debris) and continue tracking.

In the ROS 2 node wrapper, these action states are mapped to Twist control velocities:

- `<STATE_FOLLOW_LINE>` / `<STATE_IGNORE>` $\rightarrow$ Linear: `0.5` m/s, Angular: `0.0` rad/s
- `<STATE_STOP_LIGHT>` / `<STATE_STOP_OBSTACLE>` $\rightarrow$ Linear: `0.0` m/s, Angular: `0.0` rad/s

---

## Package Architecture & Directory Structure

This module is designed to decouple the model's core ML / mathematical logic from the ROS 2 node communication wrapper, ensuring you can train, run inference, or execute test sweeps without loading the ROS 2 environment or starting DDS communication.

```
src/vla/
├── package.xml                 # ROS 2 package configuration
├── setup.py                    # Python setup script mapping vla & vla_src
├── setup.cfg                   # Setup configuration file
├── README.md                   # Detailed package documentation (this file)
├── src/                        # Core ML & safety business logic (decoupled)
│   ├── __init__.py
│   ├── vla_model.py            # MLX Vision-Language-Action network, VLAAction Enum, tokenization & inference
│   ├── trainer.py              # MLX training loop (Cross-Entropy loss) and trajectory dataset loader
│   └── safety_arbiter.py       # YOLO collision avoidance & state override
├── vla/                        # ROS 2 Node wrappers
│   ├── __init__.py
│   └── vla_node.py             # ROS 2 node subscribing to camera & YOLO, publishing cmd_vel
└── tests/                      # Comprehensive test suite
    ├── __init__.py
    ├── test_vla_model.py       # Core inference, VLAAction tokens, & OOM/memory tests
    ├── test_trainer.py         # Cross-Entropy calculations & weight update validations
    ├── test_safety_arbiter.py  # Bounding box collision and state-override edge cases
    └── test_vla_node.py        # ROS 2 mock testing (timeout limits, topic drops, pub/sub)
```

---

## Installation & Setup

1. Make sure your ROS 2 Humble desktop workspace is sourced:
   ```bash
   source /opt/ros/humble/setup.zsh
   ```
2. Build the workspace using `colcon`:
   ```bash
   colcon build --packages-select vla
   ```
3. Source the local install:
   ```bash
   source install/setup.zsh
   ```

---

## Training Data Specifications

To train or fine-tune the VLA model, the dataset directory must contain trajectory episodes as structured folders containing step-by-step images and annotation logs.

### 1. File Layout

```
dataset_dir/
├── metadata.json               # Vocabulary size, total episodes configuration
├── episode_00001/
│   ├── step_000_image.jpg      # RGB Camera frame for step 0
│   ├── step_001_image.jpg      # RGB Camera frame for step 1
│   └── episode_data.json       # Navigation targets and ground truth actions
└── episode_00002/
    ├── step_000_image.jpg
    └── episode_data.json
```

### 2. `episode_data.json` Schema

Each episode requires a JSON array containing trajectory step parameters:

```json
[
  {
    "step_index": 0,
    "image_path": "step_000_image.jpg",
    "instruction": "What should the robot do given the current scene?",
    "action": "<STATE_FOLLOW_LINE>"
  }
]
```

- **`image_path`**: Relative path to the JPEG file.
- **`instruction`**: String navigation instruction command.
- **`action`**: String action state token (one of the 6 `VLAAction` Enum values).

---

## Launching & Execution

### 1. Running the VLA Controller Node

You can spin up the ROS 2 node using standard ROS 2 CLI. The node subscribes to `/camera/image_raw` (camera frames) and `/yolo/detections` (YOLO bounding boxes), and outputs control instructions on `/cmd_vel`.

```bash
ros2 run vla vla_node --ros-args -p instruction:="drive straight and turn left" -p image_timeout:=0.5
```

### 2. Starting Fine-Tuning/Training

The trainer can be run directly as a python script pointing to your training data directory:

```python
from vla_src.vla_model import VlaModel
from vla_src.trainer import VlaTrainer, DatasetLoader

model = VlaModel()
trainer = VlaTrainer(model, lr=1e-4)
loader = DatasetLoader("/path/to/dataset_dir")

for epoch in range(10):
    for i in range(len(loader)):
        image, instruction, target_action = loader.get_sample(i)
        loss = trainer.train_step(image, instruction, target_action)
        print(f"Epoch {epoch}, Step {i}, Loss: {loss:.4f}")
```

---

## Verification & Testing

Our test-driven architecture covers model structures, optimizers, parameter checks, bounding box triggers, and ROS 2 watchdog timeouts.

To run the full suite:

```bash
python3 -m pytest src/vla/tests/
```

All tests run successfully without requiring active ROS 2 DDS networks or specific GPU hardware (automatic fallbacks are active in test runtimes).
