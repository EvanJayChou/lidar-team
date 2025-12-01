"""
`- Subscribes to a *clustered* PointCloud2 topic
  - Extracts (x, y, z, cluster_id) for each point
  - Groups points by cluster_id
  - Computes a 3D bounding box for each cluster
  - Publishes RViz markers (boxes + outlines)

"""

from __future__ import annotations
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
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
            Float32MultiArray,
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
    def on_cloud(self, msg: Float32MultiArray):
        """
        Called when a clustered point cloud arrives.
        """
        #convert it into (N.4) numpy array
        raw = np.array(msg.data, dtype=np.float32)
        if raw.size == 0:
            return
        if raw.size % 4 != 0:
            self.get_logger().warn(f"Got msg.data of length {raw.size}, not divisible by 4")
            return

        data = raw.reshape(-1, 4)
        # === Create MarkerArray ===
        marker_array = MarkerArray()

        # Clear old markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for cid in np.unique(data[:,3]):
            mask = (data[:, 3] == cid)
            cluster_data = data[mask]
            clusterid=int(cid)
            max_x = np.max(cluster_data[:, 0])
            max_y = np.max(cluster_data[:, 1])
            max_z = np.max(cluster_data[:, 2])

            min_x = np.min(cluster_data[:, 0])
            min_y = np.min(cluster_data[:, 1])
            min_z = np.min(cluster_data[:, 2])

            # Center and size
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            cz = (min_z + max_z) / 2

            sx = max_x - min_x
            sy = max_y - min_y
            sz = max_z - min_z

            # === Create 3D box marker ===
            box = Marker()
            # box.header = msg.header
            box.ns = "cluster_boxes"
            box.id = clusterid
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
            # outline.header = msg.header
            outline.ns = "cluster_outlines"
            outline.id = clusterid + 10000
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