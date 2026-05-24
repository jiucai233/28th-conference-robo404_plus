import os
import json
import logging
import numpy as np
from vla_model import VlaModel, VLAAction, HAS_MLX, mx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VlaTrainer")

# Action string to integer index mapping
ACTION_TO_INDEX = {act.value: idx for idx, act in enumerate(VLAAction)}

# Import MLX optimizer modules dynamically
HAS_MLX_OPTIMIZERS = False
if HAS_MLX:
    try:
        import mlx.optimizers as optim
        HAS_MLX_OPTIMIZERS = True
    except ImportError:
        pass


class MockOptimizer:
    """Fallback optimizer implementation when MLX is not present."""
    def __init__(self, lr=1e-4):
        self.lr = lr

    def update(self, model, grads):
        # Mock parameter update in numpy
        for p_name, weight in model.model.action_head.__dict__.items():
            if isinstance(weight, np.ndarray):
                # Update weights by subtracting a dummy scaled gradient
                model.model.action_head.__dict__[p_name] -= self.lr * 0.01


class DatasetLoader:
    """Loads and parses trajectories for VLA training from a structured dataset folder."""
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.metadata = {}
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} does not exist.")

        metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

        # Walk through episode directories
        for entry in sorted(os.listdir(self.dataset_dir)):
            episode_path = os.path.join(self.dataset_dir, entry)
            if os.path.isdir(episode_path) and entry.startswith("episode_"):
                data_json_path = os.path.join(episode_path, "episode_data.json")
                if os.path.exists(data_json_path):
                    with open(data_json_path, "r") as f:
                        steps = json.load(f)
                        for step in steps:
                            img_abs_path = os.path.join(episode_path, step["image_path"])
                            
                            # Parse string action and map to class index
                            action_str = step["action"]
                            action_idx = ACTION_TO_INDEX.get(action_str, 0)
                            
                            self.samples.append({
                                "image_path": img_abs_path,
                                "instruction": step["instruction"],
                                "action_str": action_str,
                                "action_idx": action_idx
                            })
        logger.info(f"Loaded {len(self.samples)} training samples from {self.dataset_dir}")

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx: int):
        sample = self.samples[idx]
        if os.path.exists(sample["image_path"]):
            image_data = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
        return image_data, sample["instruction"], sample["action_str"]


class VlaTrainer:
    """Trainer class for training and optimization of VlaModel."""
    def __init__(self, model: VlaModel, lr: float = 1e-4):
        self.model = model
        self.lr = lr
        
        if HAS_MLX_OPTIMIZERS:
            self.optimizer = optim.Adam(learning_rate=lr)
        else:
            self.optimizer = MockOptimizer(lr=lr)

    def compute_loss(self, predicted_logits: mx.array, target_index: int) -> mx.array:
        """Computes Softmax Cross-Entropy loss with log-sum-exp trick."""
        if HAS_MLX:
            # Log-Sum-Exp Cross Entropy in MLX
            max_val = mx.max(predicted_logits)
            log_sum_exp = max_val + mx.log(mx.sum(mx.exp(predicted_logits - max_val)))
            return -predicted_logits[target_index] + log_sum_exp
        else:
            # NumPy simulation
            max_val = np.max(predicted_logits)
            log_sum_exp = max_val + np.log(np.sum(np.exp(predicted_logits - max_val)))
            return -predicted_logits[target_index] + log_sum_exp

    def train_step(self, image: np.ndarray, instruction: str, target_action: str) -> float:
        """Performs a single parameter optimization step.
        
        Args:
            image: HxWxC numpy array.
            instruction: string navigation goal.
            target_action: string discrete action state token (e.g. "<STATE_FOLLOW_LINE>").
            
        Returns:
            float: loss value of the step.
        """
        # Convert string action to target index
        target_idx = ACTION_TO_INDEX.get(target_action, 0)

        if HAS_MLX and HAS_MLX_OPTIMIZERS:
            import mlx.core as mx_real
            
            # Loss helper function
            def loss_fn(model_params, img, tokens, t_idx):
                self.model.model.update(model_params)
                logits = self.model.model(img, tokens)
                return self.compute_loss(logits, t_idx)

            image_mx = mx.array(image.astype(np.float32) / 255.0)
            tokens_mx = mx.array(self.model.tokenize(instruction))

            # Compile gradient functions
            loss_and_grad_fn = mx_real.value_and_grad(loss_fn)
            loss, grads = loss_and_grad_fn(self.model.model.parameters(), image_mx, tokens_mx, target_idx)
            
            self.optimizer.update(self.model.model, grads)
            mx_real.eval(self.model.model.parameters(), loss)
            
            loss_val = float(loss)
        else:
            # Fallback numpy simulation
            # Run model forward pass to get logits
            tokens = self.model.tokenize(instruction)
            image_mx = mx.array(image.astype(np.float32) / 255.0)
            tokens_mx = mx.array(tokens)
            logits_mx = self.model.model(image_mx, tokens_mx)
            
            loss_mx = self.compute_loss(logits_mx, target_idx)
            loss_val = float(loss_mx)
            
            # Calculate gradient (Softmax probability error: probs - target_one_hot)
            probs = np.exp(logits_mx - np.max(logits_mx))
            probs /= np.sum(probs)
            target_one_hot = np.zeros(len(VLAAction), dtype=np.float32)
            target_one_hot[target_idx] = 1.0
            
            dummy_grad = {"action_head.weight": probs - target_one_hot}
            self.optimizer.update(self.model, dummy_grad)
            
        return loss_val
