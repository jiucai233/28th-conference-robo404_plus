import logging
from vla_model import VLAAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafetyArbiter")


class SafetyArbiter:
    """Decoupled safety controller to perform collision-override checks on discrete states."""
    def __init__(
        self,
        collision_size_threshold: float = 120.0,
        center_zone_width: float = 200.0
    ):
        # Max bounding box width/height indicating target is too close in pixels
        self.collision_size_threshold = collision_size_threshold
        # Pixel width around horizontal center where obstacles are in the direct path
        self.center_zone_width = center_zone_width

    def evaluate_collision_threat(
        self,
        detections: list,
        image_width: int = 640
    ) -> bool:
        """Determines if any detected bounding boxes represent a collision hazard.
        
        Args:
            detections: List of detection dicts with schema:
                {
                    "class_name": str,
                    "x": float, (center x in pixels)
                    "y": float, (center y in pixels)
                    "w": float, (width in pixels)
                    "h": float  (height in pixels)
                }
            image_width: Resolution width of the input frame.
            
        Returns:
            bool: True if emergency braking should be triggered.
        """
        if not detections:
            return False

        center_x = image_width / 2.0
        left_boundary = center_x - (self.center_zone_width / 2.0)
        right_boundary = center_x + (self.center_zone_width / 2.0)

        for det in detections:
            # Handle corrupted or missing payload dictionary keys gracefully
            if not isinstance(det, dict):
                logger.warning(f"Invalid detection payload format ignored: {det}")
                continue
                
            try:
                class_name = det.get("class_name", "unknown")
                x = float(det["x"])
                y = float(det["y"])
                w = float(det["w"])
                h = float(det["h"])
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Malformed bounding box record ignored: {det}. Exception: {e}")
                continue

            # Check if object is in direct lane path (horizontally centered)
            in_path = left_boundary <= x <= right_boundary

            # Check if object is close (width or height exceeds size threshold)
            too_close = w >= self.collision_size_threshold or h >= self.collision_size_threshold

            # Hazard classes: people, vehicles, obstacles
            hazard_classes = {"person", "car", "truck", "stop sign", "bicycle", "obstacle"}
            is_hazard = class_name.lower() in hazard_classes or class_name.lower() == "unknown"

            if in_path and too_close and is_hazard:
                logger.warning(
                    f"Collision risk: obstacle '{class_name}' at x={x:.1f} with size {w:.1f}x{h:.1f}."
                )
                return True

        return False

    def arbitrate(
        self,
        proposed_action: VLAAction,
        detections: list,
        image_width: int = 640
    ) -> VLAAction:
        """Processes proposed actions and yields final safe values.
        
        If an obstacle threat is detected, forces action to STOP_OBSTACLE.
        """
        # Validate proposed action type
        if not isinstance(proposed_action, VLAAction):
            logger.warning(f"Invalid action token type: {proposed_action}. Fallback to FOLLOW_LINE.")
            proposed_action = VLAAction.FOLLOW_LINE

        # Assess hazards
        hazard_detected = self.evaluate_collision_threat(detections, image_width)

        if hazard_detected:
            # Safe Stop override state
            return VLAAction.STOP_OBSTACLE

        return proposed_action
