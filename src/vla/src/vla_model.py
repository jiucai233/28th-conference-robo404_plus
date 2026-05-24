import numpy as np
import logging
import gc
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VlaModel")

# Discrete Action Space States
class VLAAction(str, Enum):
    FOLLOW_LINE = "<STATE_FOLLOW_LINE>"
    STOP_LIGHT = "<STATE_STOP_LIGHT>"
    STOP_OBSTACLE = "<STATE_STOP_OBSTACLE>"
    IGNORE = "<STATE_IGNORE>"
    NAV_LEFT = "<STATE_NAV_LEFT>"
    NAV_RIGHT = "<STATE_NAV_RIGHT>"


# Decoupled Import for Apple MLX
HAS_MLX = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    logger.warning("mlx package not found. Running with NumPy fallback mock model.")
    # Mocking MLX structures for unit tests and cross-platform compatibility
    class MockMetal:
        @staticmethod
        def clear_cache():
            pass

    class MockMX:
        float32 = np.float32
        int32 = np.int32
        metal = MockMetal

        @staticmethod
        def array(data, dtype=None):
            return np.array(data, dtype=dtype)

        @staticmethod
        def eval(*args):
            pass

    class MockModule:
        def __init__(self):
            self._submodules = {}

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    mx = MockMX()
    
    class MockNN:
        Module = MockModule
        class Linear:
            def __init__(self, input_dim, output_dim):
                self.weight = np.random.randn(output_dim, input_dim).astype(np.float32)
                self.bias = np.zeros(output_dim).astype(np.float32)

            def __call__(self, x):
                return x @ self.weight.T + self.bias

    nn = MockNN()


class SimpleTokenizer:
    """A lightweight, deterministic word-level tokenizer for navigation instructions."""
    def __init__(self, max_length=16):
        self.max_length = max_length
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "forward": 2,
            "straight": 3,
            "turn": 4,
            "left": 5,
            "right": 6,
            "stop": 7,
            "slow": 8,
            "speed": 9,
            "up": 10,
            "down": 11,
            "and": 12,
            "at": 13,
            "the": 14,
            "intersection": 15
        }
        self.vocab_size = len(self.vocab)

    def tokenize(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            return np.zeros(self.max_length, dtype=np.int32)
        
        words = text.lower().replace(",", "").replace(".", "").split()
        tokens = []
        for word in words:
            tokens.append(self.vocab.get(word, self.vocab["<unk>"]))
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens += [self.vocab["<pad>"]] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            
        return np.array(tokens, dtype=np.int32)


class MLXModelImpl(nn.Module):
    """The internal neural network structure built using Apple MLX."""
    def __init__(self, vocab_size: int, embed_dim: int = 16, action_dim: int = 6):
        super().__init__()
        # Visual projection: flatten 224x224x3 image projection layer
        self.image_proj = nn.Linear(224 * 224 * 3, embed_dim)
        # Instruction embedding layer (mocked/linear projection for demonstration)
        self.text_proj = nn.Linear(16, embed_dim)
        # Action decoder head (outputs logits for 6 discrete action states)
        self.action_head = nn.Linear(embed_dim * 2, action_dim)

    def __call__(self, image: mx.array, tokens: mx.array) -> mx.array:
        # Flatten image
        flat_image = mx.array(image.reshape(-1).astype(mx.float32))
        img_feats = self.image_proj(flat_image)
        
        # Token feature extraction
        flat_tokens = mx.array(tokens.astype(mx.float32))
        text_feats = self.text_proj(flat_tokens)
        
        # Concatenate features
        if HAS_MLX:
            import mlx.core as mx_real
            fused = mx_real.concatenate([img_feats, text_feats], axis=0)
        else:
            fused = np.concatenate([img_feats, text_feats], axis=0)
            
        action_logits = self.action_head(fused)
        return action_logits


class VlaModel:
    """Wrapper that exposes inference and training endpoints for the VLA architecture."""
    def __init__(self, embed_dim: int = 16):
        self.tokenizer = SimpleTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.action_dim = len(VLAAction)
        self.embed_dim = embed_dim
        
        # Instantiate MLX model parameters with action_dim=6
        self.model = MLXModelImpl(self.vocab_size, embed_dim, self.action_dim)
        self.actions_list = list(VLAAction)
        
    def tokenize(self, instruction: str) -> np.ndarray:
        return self.tokenizer.tokenize(instruction)

    def predict_action(self, image: np.ndarray, instruction: str) -> VLAAction:
        """Runs a forward inference pass.
        
        Args:
            image: numpy array of shape (H, W, C).
            instruction: natural language navigation string.
        
        Returns:
            VLAAction: Discrete action state token.
        """
        # Validate inputs
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a non-empty numpy array.")
            
        # Expect H=224, W=224, C=3 for the core model structure
        if image.shape != (224, 224, 3):
            logger.warning(f"Resizing input image from {image.shape} to (224, 224, 3)")
            resized_image = np.zeros((224, 224, 3), dtype=np.float32)
            h, w = min(image.shape[0], 224), min(image.shape[1], 224)
            c = min(image.shape[2], 3) if len(image.shape) > 2 else 1
            resized_image[:h, :w, :c] = image[:h, :w, :c]
            image = resized_image

        tokens = self.tokenize(instruction)
        
        # Convert arrays to MLX formats
        image_mx = mx.array(image.astype(np.float32) / 255.0)
        tokens_mx = mx.array(tokens)
        
        # Execute MLX graph forward pass to get action logits (shape: [6])
        logits_mx = self.model(image_mx, tokens_mx)
        
        # Forces computation in MLX (OOM prevention & validation)
        mx.eval(logits_mx)
        
        # Convert output back to numpy array
        if HAS_MLX:
            logits_np = np.array(logits_mx)
        else:
            logits_np = logits_mx # In fallback, it is already a numpy array
            
        # Verify output integrity
        if np.isnan(logits_np).any():
            logger.error("VLA Model predicted NaN logits. Defaulting to FOLLOW_LINE.")
            return VLAAction.FOLLOW_LINE

        # Argmax to find the predicted action index
        pred_idx = int(np.argmax(logits_np))
        
        # Safe lookup with fallback check
        if 0 <= pred_idx < len(self.actions_list):
            selected_action = self.actions_list[pred_idx]
        else:
            logger.warning(f"Predicted index {pred_idx} out of range. Defaulting to FOLLOW_LINE.")
            selected_action = VLAAction.FOLLOW_LINE

        # Clear memory immediately to optimize resource usage on device
        self.clear_memory()
        
        return selected_action

    def clear_memory(self):
        """Forces MLX metal memory cache release and garbage collection to prevent OOM."""
        mx.metal.clear_cache()
        gc.collect()
