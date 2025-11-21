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
    
        try:
            pts = point_cloud2.read_points(
                msg,
                field_names=["x", "y", "z", self.cluster_field],
                skip_nans=True
            )
        except Exception as e:
            self.get_logger().error(f"Field '{self.cluster_field}' not found in cloud")
            return

        # ------------------------------------------------------------
        # 1) Read into NumPy arrays (vector-friendly)
        # ------------------------------------------------------------
        xyz_list = []
        cid_list = []

        for (x, y, z, cid) in pts:
            xyz_list.append((x, y, z))
            cid_list.append(int(cid))

        if not xyz_list:
            return

        # (N, 3) float32 array of xyz
        xyz = np.asarray(xyz_list, dtype=np.float32)
        # (N,) int32 array of cluster ids
        labels = np.asarray(cid_list, dtype=np.int32)

        # ------------------------------------------------------------
        # 2) Unique cluster IDs + stats in one shot
        # ------------------------------------------------------------
        # unique_cids: sorted unique cluster IDs, shape (K,)
        # inverse: for each point i, the index k such that unique_cids[k] == labels[i]
        # counts: number of points per cluster, shape (K,)
        unique_cids, inverse, counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )

        # ------------------------------------------------------------
        # 3) Prepare MarkerArray and clear old markers
        # ------------------------------------------------------------
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        # ------------------------------------------------------------
        # 4) Loop over clusters (now K clusters instead of N points)
        # ------------------------------------------------------------
        for cluster_idx, (cid, count) in enumerate(zip(unique_cids, counts)):

            # Skip small clusters
            if count < self.min_pts:
                continue

            # Boolean mask: True for points belonging to this cluster
            mask = (inverse == cluster_idx)
            pts_c = xyz[mask]        # shape (count, 3)

            # Vectorized min/max over x,y,z
            mins = pts_c.min(axis=0)  # [min_x, min_y, min_z]
            maxs = pts_c.max(axis=0)  # [max_x, max_y, max_z]

            # Center and size: still fully vectorized
            center = 0.5 * (mins + maxs)    # [cx, cy, cz]
            size   = maxs - mins            # [sx, sy, sz]

            min_x, min_y, min_z = mins
            max_x, max_y, max_z = maxs
            cx, cy, cz = center
            sx, sy, sz = size

            # --------------------------------------------------------
            # 4a) 3D box marker
            # --------------------------------------------------------
            box = Marker()
            box.header = msg.header
            box.ns = "cluster_boxes"
            box.id = int(cid)
            box.type = Marker.CUBE
            box.action = Marker.ADD

            box.pose.position.x = float(cx)
            box.pose.position.y = float(cy)
            box.pose.position.z = float(cz)

            box.scale.x = float(max(sx, 0.01))
            box.scale.y = float(max(sy, 0.01))
            box.scale.z = float(max(sz, 0.01))

            box.color.r = 1.0
            box.color.g = 1.0
            box.color.b = 0.0
            box.color.a = 0.35

            marker_array.markers.append(box)

            # --------------------------------------------------------
            # 4b) 2D footprint outline
            # --------------------------------------------------------
            outline = Marker()
            outline.header = msg.header
            outline.ns = "cluster_outlines"
            outline.id = int(cid) + 10000
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.scale.x = 0.06  # line thickness [m]

            outline.color.r = 0.0
            outline.color.g = 1.0
            outline.color.b = 0.0
            outline.color.a = 1.0

            z_draw = float(min_z)

            corners = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            ]

            for px, py in corners:
                p = Point()
                p.x = float(px)
                p.y = float(py)
                p.z = z_draw
                outline.points.append(p)

            marker_array.markers.append(outline)

        # ------------------------------------------------------------
        # 5) Publish all boxes
        # ------------------------------------------------------------
        self.pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterBoundingBoxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()