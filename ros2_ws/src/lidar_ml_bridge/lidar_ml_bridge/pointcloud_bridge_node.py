"""ROS 2 node for subscribing to a PointCloud2 topic and delegating parsing to
pure utility functions (see `pointcloud_parser.py`). Keeps ROS concerns (params,
QoS, logging, subscription) separated from data extraction logic.

Usage:
    ros2 run lidar_ml_bridge lidar_pointcloud_bridge --ros-args -p pointcloud_topic:=/lidar/points

Parameters:
    pointcloud_topic (string): Topic name of PointCloud2. Default: /lidar_points
    use_numpy (bool): Use NumPy backend if available. Default: True
    keep_fields (string): Comma-separated subset of fields to keep (e.g. "x,y,z,intensity"). Empty => auto.

Extension:
    Override `send_to_ml` to forward `ParsedCloud` objects downstream.
"""

from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2

from .pointcloud_parser import ParsedCloud, parse_pointcloud, XYZ_FIELDS  # re-exported symbols


class PointCloudBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_bridge")

        self.declare_parameter("pointcloud_topic", "/lidar_points")
        self.declare_parameter("use_numpy", True)
        self.declare_parameter("keep_fields", "")

        topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        self._use_numpy = self.get_parameter("use_numpy").get_parameter_value().bool_value
        keep_fields_raw = self.get_parameter("keep_fields").get_parameter_value().string_value
        self._keep_fields = [f.strip() for f in keep_fields_raw.split(",") if f.strip()] if keep_fields_raw else []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._subscription = self.create_subscription(
            PointCloud2,
            topic,
            self._on_pointcloud,
            qos,
        )
        self.get_logger().info(f"Subscribed to PointCloud2 topic: {topic}")

    # -------------------------- Callback --------------------------
    def _on_pointcloud(self, msg: PointCloud2) -> None:
        parsed = parse_pointcloud(
            msg,
            keep_fields=self._keep_fields if self._keep_fields else None,
            use_numpy=self._use_numpy,
        )
        # Placeholder for forwarding to ML pipeline
        self.send_to_ml(parsed)

    # -------------------------- Extension Hook -------------------
    def send_to_ml(self, parsed: ParsedCloud) -> None:  # pragma: no cover - skeleton hook
        """Override or extend: forward parsed data into an ML pipeline.
        Examples:
          - Publish a custom message
          - Push onto a multiprocessing queue
          - Perform pre-processing and batching
        Currently logs a summary only.
        """
        num_points = len(parsed.xyz)
        self.get_logger().debug(
            f"Parsed cloud frame={parsed.frame_id} points={num_points} has_intensity={'yes' if parsed.intensity is not None else 'no'}"
        )


def main(args=None):  # pragma: no cover
    rclpy.init(args=args)
    node = PointCloudBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down pointcloud bridge node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
