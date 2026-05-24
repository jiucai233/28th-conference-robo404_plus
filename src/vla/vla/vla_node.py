import sys
import os
import time
import numpy as np

# Configure fallback imports for cross-platform/non-ROS CI runs
HAS_ROS2 = True
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from yolo_msgs.msg import DetectionArray
except ImportError:
    HAS_ROS2 = False
    # Mock ROS 2 base classes for test environments
    class Node:
        def __init__(self, name):
            self.name = name
            self._parameters = {}
        def create_subscription(self, msg_type, topic, callback, qos):
            return None
        def create_publisher(self, msg_type, topic, qos):
            class MockPublisher:
                def publish(self, msg):
                    pass
            return MockPublisher()
        def create_timer(self, period, callback):
            return None
        def declare_parameter(self, name, default):
            self._parameters[name] = default
        def get_parameter(self, name):
            val = self._parameters.get(name, None)
            if val is None:
                if name == 'instruction':
                    val = 'drive straight'
                elif name == 'image_timeout':
                    val = 0.5
            class MockParam:
                value = val
            return MockParam()
        def get_clock(self):
            class MockClock:
                def now(self):
                    class MockTime:
                        nanoseconds = int(time.time() * 1e9)
                    return MockTime()
            return MockClock()
        def get_logger(self):
            class MockLogger:
                def info(self, msg):
                    pass
                def warning(self, msg):
                    pass
                def error(self, msg):
                    pass
            return MockLogger()
            
    class Image:
        pass
    class Twist:
        pass
    class DetectionArray:
        pass

HAS_CV_BRIDGE = True
try:
    from cv_bridge import CvBridge
except ImportError:
    HAS_CV_BRIDGE = False

# Import VLA model and Safety Arbiter from core logic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from vla_model import VlaModel, VLAAction
from safety_arbiter import SafetyArbiter

# Action state to Twist velocity mapping profile
ACTION_VELOCITY_MAP = {
    VLAAction.FOLLOW_LINE: (0.5, 0.0),
    VLAAction.IGNORE: (0.5, 0.0),
    VLAAction.STOP_LIGHT: (0.0, 0.0),
    VLAAction.STOP_OBSTACLE: (0.0, 0.0),
    VLAAction.NAV_LEFT: (0.2, 0.5),
    VLAAction.NAV_RIGHT: (0.2, -0.5),
}


class VlaNode(Node):
    """ROS 2 Humble Node wrapping the MLX VLA model and safety arbiter controls."""
    def __init__(self):
        super().__init__('vla_node')
        
        # Declare parameters
        self.declare_parameter('instruction', 'drive straight')
        self.declare_parameter('image_timeout', 0.5)  # seconds
        
        # Instantiate core MLX VLA and Safety Arbiter components
        self.vla_model = VlaModel()
        self.safety_arbiter = SafetyArbiter()
        
        if HAS_CV_BRIDGE:
            self.bridge = CvBridge()
        else:
            self.bridge = None

        # Data states
        self.latest_image = None
        self.last_image_time = None
        self.latest_detections = []

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.yolo_sub = self.create_subscription(
            DetectionArray,
            '/yolo/detections',
            self.yolo_callback,
            10
        )

        # 10Hz control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("VLA ROS 2 Node initialized successfully.")

    def image_callback(self, msg: Image):
        """Processes incoming camera images."""
        self.last_image_time = self.get_clock().now()
        
        # Image conversion using cv_bridge or pure numpy fallback
        if self.bridge:
            try:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            except Exception as e:
                self.get_logger().error(f"cv_bridge image conversion failed: {e}")
                self.latest_image = self.numpy_image_fallback(msg)
        else:
            self.latest_image = self.numpy_image_fallback(msg)

    def numpy_image_fallback(self, msg: Image) -> np.ndarray:
        """Parses image message byte buffer directly in standard numpy format."""
        try:
            if not hasattr(msg, 'data') or not hasattr(msg, 'height') or not hasattr(msg, 'width'):
                return None
            
            channels = 3
            img_arr = np.frombuffer(msg.data, dtype=np.uint8)
            expected_size = msg.height * msg.width * channels
            if len(img_arr) >= expected_size:
                return img_arr[:expected_size].reshape((msg.height, msg.width, channels))
        except Exception as e:
            self.get_logger().error(f"Numpy direct image conversion fallback failed: {e}")
        return None

    def yolo_callback(self, msg: DetectionArray):
        """Processes and formats incoming YOLO bounding boxes."""
        parsed = []
        try:
            if hasattr(msg, 'detections'):
                for det in msg.detections:
                    center_x = det.bbox.center.position.x
                    center_y = det.bbox.center.position.y
                    width = det.bbox.size.x
                    height = det.bbox.size.y
                    
                    parsed.append({
                        "class_name": det.class_name,
                        "x": center_x,
                        "y": center_y,
                        "w": width,
                        "h": height
                    })
            self.latest_detections = parsed
        except Exception as e:
            self.get_logger().warning(f"Error parsing YOLO DetectionArray: {e}")

    def control_loop(self):
        """Evaluation and velocity publishing tick."""
        # Create empty Twist command msg
        cmd_msg = Twist() if HAS_ROS2 else type('MockTwist', (), {'linear': type('Vector3', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})(), 'angular': type('Vector3', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()})()
        
        # Get active parameters
        instruction = self.get_parameter('instruction').value
        timeout_limit = self.get_parameter('image_timeout').value

        # Watchdog logic checking for topic drops/network timeouts
        now = self.get_clock().now()
        if self.last_image_time is None or self.latest_image is None:
            self.publish_zero_velocity(cmd_msg)
            return

        elapsed_sec = (now.nanoseconds - self.last_image_time.nanoseconds) / 1e9

        if elapsed_sec > timeout_limit:
            self.get_logger().warning(
                f"Camera topic dropped for {elapsed_sec:.2f}s. Triggering safe stop deceleration."
            )
            self.publish_zero_velocity(cmd_msg)
            return

        try:
            # 1. Run core VLA MLX model inference -> returns VLAAction enum
            proposed_action = self.vla_model.predict_action(self.latest_image, instruction)

            # 2. Arbitrate actions using bounding box constraints -> returns VLAAction enum
            safe_action = self.safety_arbiter.arbitrate(
                proposed_action,
                self.latest_detections
            )

            # 3. Map action state to physical Twist velocities
            linear_v, angular_w = ACTION_VELOCITY_MAP.get(safe_action, (0.0, 0.0))

            # 4. Publish control commands
            cmd_msg.linear.x = linear_v
            cmd_msg.angular.z = angular_w
            self.cmd_vel_pub.publish(cmd_msg)

        except Exception as e:
            self.get_logger().error(f"Error in control loop execution: {e}")
            self.publish_zero_velocity(cmd_msg)

    def publish_zero_velocity(self, cmd_msg):
        """Forces velocities to zero and publishes Twist to vehicle."""
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    if not HAS_ROS2:
        print("Cannot run node without full ROS 2 installation.")
        return
        
    rclpy.init(args=args)
    node = VlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
