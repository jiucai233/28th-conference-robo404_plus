import os
import sys
import pytest

# Dynamically patch python path to allow importing core logic files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from safety_arbiter import SafetyArbiter
from vla_model import VLAAction

def test_empty_or_malformed_detections():
    arbiter = SafetyArbiter()
    
    # None detections
    assert not arbiter.evaluate_collision_threat(None)
    
    # Empty list
    assert not arbiter.evaluate_collision_threat([])
    
    # Malformed inputs
    assert not arbiter.evaluate_collision_threat(["invalid_item", 42])
    
    malformed = [
        {"class_name": "person", "x": "corrupted", "y": 240, "w": 100, "h": 100},
        {"class_name": "car", "y": 240, "w": 100}
    ]
    assert not arbiter.evaluate_collision_threat(malformed)

def test_collision_threat_detection():
    arbiter = SafetyArbiter(collision_size_threshold=100.0, center_zone_width=200.0)
    
    # Obstacle is in center zone (320 is center) and too close
    hazard = [
        {"class_name": "person", "x": 320.0, "y": 240.0, "w": 120.0, "h": 150.0}
    ]
    assert arbiter.evaluate_collision_threat(hazard, image_width=640)

    # Obstacle is small (far away)
    far_hazard = [
        {"class_name": "person", "x": 320.0, "y": 240.0, "w": 40.0, "h": 50.0}
    ]
    assert not arbiter.evaluate_collision_threat(far_hazard, image_width=640)

    # Obstacle is off to the side
    side_hazard = [
        {"class_name": "person", "x": 100.0, "y": 240.0, "w": 150.0, "h": 150.0}
    ]
    assert not arbiter.evaluate_collision_threat(side_hazard, image_width=640)

def test_arbitrate_safety_override():
    arbiter = SafetyArbiter(collision_size_threshold=100.0)
    
    # Safe path: should pass proposed action token unchanged
    detections_safe = [
        {"class_name": "person", "x": 100.0, "y": 240.0, "w": 150.0, "h": 150.0}
    ]
    action = arbiter.arbitrate(VLAAction.NAV_LEFT, detections_safe)
    assert action == VLAAction.NAV_LEFT
    
    # Unsafe path: should override proposed action to STOP_OBSTACLE
    detections_unsafe = [
        {"class_name": "car", "x": 310.0, "y": 240.0, "w": 120.0, "h": 110.0}
    ]
    action_override = arbiter.arbitrate(VLAAction.FOLLOW_LINE, detections_unsafe)
    assert action_override == VLAAction.STOP_OBSTACLE

def test_arbitrate_fallback_invalid_type():
    arbiter = SafetyArbiter()
    
    # Invalid action types should default to FOLLOW_LINE
    action = arbiter.arbitrate("invalid_string_action", [])
    assert action == VLAAction.FOLLOW_LINE
