import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Dynamically patch python path to allow importing core and node logic files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../vla')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vla_node
from vla_node import VlaNode, VLAAction

# Mock message objects for test runs
class MockPoint2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockPose2D:
    def __init__(self, x, y):
        self.position = MockPoint2D(x, y)

class MockVector2:
    def __init__(self, w, h):
        self.x = w
        self.y = h

class MockBoundingBox2D:
    def __init__(self, x, y, w, h):
        self.center = MockPose2D(x, y)
        self.size = MockVector2(w, h)

class MockDetection:
    def __init__(self, class_name, x, y, w, h):
        self.class_name = class_name
        self.bbox = MockBoundingBox2D(x, y, w, h)

class MockDetectionArray:
    def __init__(self, detections_list):
        self.detections = detections_list

class MockImageMessage:
    def __init__(self, height, width, data):
        self.height = height
        self.width = width
        self.data = data


def test_image_callback_updates():
    node = VlaNode()
    
    # 2x2 RGB Image bytes
    raw_data = bytes([10, 20, 30] * 4)
    msg = MockImageMessage(height=2, width=2, data=raw_data)
    
    node.image_callback(msg)
    
    assert node.latest_image is not None
    assert node.latest_image.shape == (2, 2, 3)
    assert node.last_image_time is not None

def test_detection_callback_parsing():
    node = VlaNode()
    
    det1 = MockDetection("person", 320.0, 240.0, 80.0, 90.0)
    det2 = MockDetection("car", 100.0, 240.0, 150.0, 130.0)
    msg = MockDetectionArray([det1, det2])
    
    node.yolo_callback(msg)
    
    assert len(node.latest_detections) == 2
    assert node.latest_detections[0]["class_name"] == "person"
    assert node.latest_detections[0]["x"] == 320.0
    assert node.latest_detections[0]["w"] == 80.0
    assert node.latest_detections[1]["class_name"] == "car"
    assert node.latest_detections[1]["x"] == 100.0
    assert node.latest_detections[1]["w"] == 150.0

def test_control_loop_publish_command():
    node = VlaNode()
    
    # Initialize state
    node.latest_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Mock clock time
    mock_now = MagicMock()
    mock_now.nanoseconds = int(100.0 * 1e9) # 100s
    node.get_clock = MagicMock()
    node.get_clock().now.return_value = mock_now
    
    # Image received just now
    mock_recv_time = MagicMock()
    mock_recv_time.nanoseconds = int(99.8 * 1e9) # 99.8s (elapsed = 0.2s < timeout = 0.5s)
    node.last_image_time = mock_recv_time
    
    # Mock model predict action to return NAV_RIGHT state
    node.vla_model.predict_action = MagicMock(return_value=VLAAction.NAV_RIGHT)
    
    # Mock publisher publish method
    node.cmd_vel_pub.publish = MagicMock()
    
    # Execute loop tick
    node.control_loop()
    
    # Verify predict_action and publisher were called
    node.vla_model.predict_action.assert_called_once_with(node.latest_image, 'drive straight')
    node.cmd_vel_pub.publish.assert_called_once()
    
    # Assert Twist message components (mapping NAV_RIGHT to linear=0.2, angular=-0.5)
    published_msg = node.cmd_vel_pub.publish.call_args[0][0]
    assert published_msg.linear.x == 0.2
    assert published_msg.angular.z == -0.5

def test_watchdog_timeout_safety_override():
    node = VlaNode()
    node.latest_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    node.cmd_vel_pub.publish = MagicMock()
    
    # Set simulated clock
    mock_now = MagicMock()
    mock_now.nanoseconds = int(100.0 * 1e9)
    node.get_clock = MagicMock()
    node.get_clock().now.return_value = mock_now

    # Case 1: No image has ever been received
    node.last_image_time = None
    node.control_loop()
    published_msg = node.cmd_vel_pub.publish.call_args[0][0]
    assert published_msg.linear.x == 0.0
    assert published_msg.angular.z == 0.0

    # Case 2: Image is too old (elapsed = 0.6s > timeout = 0.5s)
    node.cmd_vel_pub.publish.reset_mock()
    mock_recv_time = MagicMock()
    mock_recv_time.nanoseconds = int(99.4 * 1e9) # 0.6s ago
    node.last_image_time = mock_recv_time
    
    node.control_loop()
    published_msg = node.cmd_vel_pub.publish.call_args[0][0]
    assert published_msg.linear.x == 0.0
    assert published_msg.angular.z == 0.0

def test_malformed_yolo_msg_failsafe():
    node = VlaNode()
    
    # Feed completely invalid data type to yolo callback
    node.yolo_callback(None)
    
    # Should not crash, and should keep detections list empty or clean
    assert node.latest_detections == []
