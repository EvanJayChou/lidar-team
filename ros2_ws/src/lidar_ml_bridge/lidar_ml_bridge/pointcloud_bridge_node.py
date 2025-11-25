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
from std_msgs.msg import Float32MultiArray
from sklearn.cluster import KMeans
import numpy as np

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
        self._publisher = self.create_publisher(Float32MultiArray, 'clustered_cloud', 10)
        self.get_logger().info(f"Subscribed to PointCloud2 topic: {topic}")

    # -------------------------- Callback --------------------------
    def _on_pointcloud(self, msg: PointCloud2) -> None:
        parsed = parse_pointcloud(
            msg,
            keep_fields=self._keep_fields if self._keep_fields else None,
            use_numpy=self._use_numpy,
        )
        # Run K-means clustering
        clustered_points = self.cluster(parsed)
        # Create a Float32MultiArray message but TODO look into using numpy messages for added speed
        msg = Float32MultiArray()
        # Flatten the NumPy array and assign it to the data field
        msg.data = clustered_points.flatten().tolist() 

        self.pub.publish(msg)

    # -------------------------- Extension Hook -------------------
    def cluster(self, parsed: ParsedCloud) -> None:  # pragma: no cover - skeleton hook
        """
        Perform K-Means Clustering and prepare points to be republished.
        """
        # Currently unsure if we need to do any conversion on parsed.xyz to get it into a format for Kmeans
        points = parsed.xyz
        
        # Remove NaNs
        points = points[~np.isnan(points)]
        self.get_logger().info(
            f"frame {parsed.frame_id}: points={num_points} points: {points}"
        )
        num_points = len(points)
        # TODO: make this configuarable
        num_clusters = 10

        # Do Kmeans
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
        kmeans.fit(points)

        # Reshape points into nx4 array as [x,y,z, cluster_id]
        clustered_points = np.zeros(shape=(num_points, 4))
        for i, point in enumerate(points):
            clustered_points[i, 0:3] = points[i, :] 
            clustered_points[i, -1:] = kmeans.labels_[i]
        self.get_logger().info(
            f"frame {parsed.frame_id}: points={num_points} clusted_points: {clustered_points}"
        )
        return clustered_points


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
