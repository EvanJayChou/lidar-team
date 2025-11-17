"""
`- Subscribes to a *clustered* PointCloud2 topic
  - Extracts (x, y, z, cluster_id) for each point
  - Groups points by cluster_id
  - Computes a 3D bounding box for each cluster
  - Publishes RViz markers (boxes + outlines)

"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class ClusterBoundingBoxNode(Node):
    def __init__(self):
        super().__init__("cluster_bounding_box_node")

        # === Parameters ===
        self.declare_parameter("input_topic", "/clustered_cloud")
        self.declare_parameter("cluster_field", "cluster_id")
        self.declare_parameter("marker_topic", "/cluster_bounding_boxes")
        self.declare_parameter("min_points", 5)

        input_topic = self.get_parameter("input_topic").value
        self.cluster_field = self.get_parameter("cluster_field").value
        marker_topic = self.get_parameter("marker_topic").value
        self.min_pts = self.get_parameter("min_points").value

        # --- Subscribers & Publishers ---
        self.sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.on_cloud,
            10
        )

        self.pub = self.create_publisher(
            MarkerArray,
            marker_topic,
            10
        )

        self.get_logger().info(
            f"[BBOX NODE] Subscribing to {input_topic}, expecting field '{self.cluster_field}'"
        )

    # =========================================================================
    #                             MAIN CALLBACK
    # =========================================================================
    def on_cloud(self, msg: PointCloud2):
        """
        Called when a clustered point cloud arrives.
        We expect fields: x, y, z, <cluster_id>
        """
        try:
            pts = point_cloud2.read_points(
                msg,
                field_names=["x", "y", "z", self.cluster_field],
                skip_nans=True
            )
        except Exception as e:
            self.get_logger().error(f"Field '{self.cluster_field}' not found in cloud")
            return

        xyz_list = []
        cid_list = []

        for (x, y, z, cid) in pts:
            xyz_list.append([x, y, z])
            cid_list.append(int(cid))

        if len(xyz_list) == 0:
            return

        xyz = np.array(xyz_list)
        labels = np.array(cid_list)

        # === Group points by cluster ID ===
        clusters: Dict[int, np.ndarray] = {}
        for idx, cid in enumerate(labels):
            clusters.setdefault(cid, []).append(idx)

        # === Create MarkerArray ===
        marker_array = MarkerArray()

        # Clear old markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        # === For each cluster, compute bounding box ===
        for cid, idx_list in clusters.items():

            if len(idx_list) < self.min_pts:
                continue

            pts_c = xyz[idx_list]

            xs, ys, zs = pts_c[:,0], pts_c[:,1], pts_c[:,2]
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            min_z, max_z = zs.min(), zs.max()

            # Center and size
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            cz = (min_z + max_z) / 2

            sx = max_x - min_x
            sy = max_y - min_y
            sz = max_z - min_z

            # === Create 3D box marker ===
            box = Marker()
            box.header = msg.header
            box.ns = "cluster_boxes"
            box.id = cid
            box.type = Marker.CUBE
            box.action = Marker.ADD

            box.pose.position.x = cx
            box.pose.position.y = cy
            box.pose.position.z = cz

            box.scale.x = max(sx, 0.01)
            box.scale.y = max(sy, 0.01)
            box.scale.z = max(sz, 0.01)

            box.color.r = 1.0
            box.color.g = 1.0
            box.color.b = 0.0
            box.color.a = 0.35

            marker_array.markers.append(box)

            # === Outline (2D footprint) ===
            outline = Marker()
            outline.header = msg.header
            outline.ns = "cluster_outlines"
            outline.id = cid + 10000
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.scale.x = 0.06  # line thickness

            outline.color.r = 0.0
            outline.color.g = 1.0
            outline.color.b = 0.0
            outline.color.a = 1.0

            z_draw = min_z  # outline height

            corners = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            ]

            for px, py in corners:
                p = Point()
                p.x = px
                p.y = py
                p.z = z_draw
                outline.points.append(p)

            marker_array.markers.append(outline)

        # Publish all boxes
        self.pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterBoundingBoxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()