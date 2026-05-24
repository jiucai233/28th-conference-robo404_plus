import os
import sys
import json
import pytest
import numpy as np

# Dynamically patch python path to allow importing core logic files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vla_model import VlaModel, VLAAction
from trainer import DatasetLoader, VlaTrainer, ACTION_TO_INDEX

def test_dataset_loader_discrete(tmpdir):
    # Set up dummy trajectory dataset directory structure
    dataset_dir = tmpdir.mkdir("dummy_dataset")
    
    # Create metadata.json
    metadata = {"vocab_size": 16, "num_episodes": 2}
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    # Episode 1
    ep1_dir = dataset_dir.mkdir("episode_00001")
    ep1_data = [
        {"step_index": 0, "image_path": "step_000_image.jpg", "instruction": "turn left", "action": "<STATE_NAV_LEFT>"},
        {"step_index": 1, "image_path": "step_001_image.jpg", "instruction": "turn left", "action": "<STATE_NAV_LEFT>"}
    ]
    with open(os.path.join(ep1_dir, "episode_data.json"), "w") as f:
        json.dump(ep1_data, f)
        
    # Create dummy empty images
    open(os.path.join(ep1_dir, "step_000_image.jpg"), "w").close()
    open(os.path.join(ep1_dir, "step_001_image.jpg"), "w").close()

    # Episode 2
    ep2_dir = dataset_dir.mkdir("episode_00002")
    ep2_data = [
        {"step_index": 0, "image_path": "step_000_image.jpg", "instruction": "stop", "action": "<STATE_STOP_OBSTACLE>"}
    ]
    with open(os.path.join(ep2_dir, "episode_data.json"), "w") as f:
        json.dump(ep2_data, f)
    open(os.path.join(ep2_dir, "step_000_image.jpg"), "w").close()

    # Load dataset
    loader = DatasetLoader(str(dataset_dir))
    assert len(loader) == 3
    
    # Load step sample
    image, instruction, action_str = loader.get_sample(0)
    assert image.shape == (224, 224, 3)
    assert instruction == "turn left"
    assert action_str == "<STATE_NAV_LEFT>"
    
    # Verify index mapping
    assert loader.samples[0]["action_idx"] == ACTION_TO_INDEX["<STATE_NAV_LEFT>"]
    assert loader.samples[2]["action_idx"] == ACTION_TO_INDEX["<STATE_STOP_OBSTACLE>"]

def test_trainer_loss_computation():
    model = VlaModel()
    trainer = VlaTrainer(model)
    
    # Feed prediction and target index (0-5)
    logits = np.array([1.0, 2.0, 0.5, -1.0, 0.0, 1.5], dtype=np.float32)
    target_idx = 1 # corresponding to index of 2.0
    
    loss = trainer.compute_loss(logits, target_idx)
    
    # Expected Log-Sum-Exp computation
    max_val = np.max(logits)
    expected_lse = max_val + np.log(np.sum(np.exp(logits - max_val)))
    expected_loss = -logits[target_idx] + expected_lse
    
    assert isinstance(loss, (float, np.floating))
    assert np.isclose(loss, expected_loss)

def test_trainer_step_and_weight_update():
    model = VlaModel()
    trainer = VlaTrainer(model, lr=0.1)
    
    # Record initial weights of projection heads
    initial_weights = np.copy(model.model.action_head.weight)
    
    # Inputs
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    instruction = "turn left"
    target = "<STATE_NAV_LEFT>"
    
    # Run training step
    loss = trainer.train_step(image, instruction, target)
    
    assert loss >= 0.0
    
    # Verify parameter changes (weights are updated)
    updated_weights = model.model.action_head.weight
    assert not np.array_equal(initial_weights, updated_weights)
