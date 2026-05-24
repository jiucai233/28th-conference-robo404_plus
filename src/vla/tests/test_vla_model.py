import os
import sys
import pytest
import numpy as np

# Dynamically patch python path to allow importing core logic files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vla_model import VlaModel, SimpleTokenizer, VLAAction

def test_tokenizer_parsing():
    tokenizer = SimpleTokenizer(max_length=10)
    
    # Standard navigation commands
    tokens = tokenizer.tokenize("turn left at the intersection")
    assert len(tokens) == 10
    assert tokens[0] == tokenizer.vocab["turn"]
    assert tokens[1] == tokenizer.vocab["left"]
    
    # Empty string input
    empty_tokens = tokenizer.tokenize("")
    assert len(empty_tokens) == 10
    assert np.all(empty_tokens == 0)
    
    # Unknown vocabulary words
    unknown_tokens = tokenizer.tokenize("xyz unusual command")
    assert unknown_tokens[0] == tokenizer.vocab["<unk>"]
    assert unknown_tokens[1] == tokenizer.vocab["<unk>"]

def test_vla_model_discrete_inference():
    model = VlaModel()
    
    # Check valid inputs
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    action = model.predict_action(dummy_image, "forward straight")
    
    assert isinstance(action, VLAAction)
    assert action in VLAAction
    
    # Non-conforming image shapes should be auto-resized/sliced gracefully
    non_conforming_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    action_resized = model.predict_action(non_conforming_image, "stop")
    assert isinstance(action_resized, VLAAction)

def test_invalid_vla_model_inputs():
    model = VlaModel()
    
    # Null image check
    with pytest.raises(ValueError):
        model.predict_action(None, "stop")

def test_nan_prediction_fallback():
    model = VlaModel()
    
    # Force model weight or output evaluation to generate NaN
    # to test the model's fallback recovery
    model.model.action_head.weight = np.full_like(
        model.model.action_head.weight, np.nan
    )
    
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    action = model.predict_action(dummy_image, "forward")
    
    # Should fall back to FOLLOW_LINE
    assert action == VLAAction.FOLLOW_LINE

def test_memory_clearance_execution():
    model = VlaModel()
    
    # Verify we can invoke memory clear without throwing exceptions
    model.clear_memory()
    assert True
