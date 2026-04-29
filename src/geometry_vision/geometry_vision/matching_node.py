

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from surveillance_interfaces.msg import DescriptorArray, MatchArray


class FeatureMatchingNode(Node):

    def __init__(self):
        super().__init__('feature_matching_node')

        self.declare_parameter('match_threshold', 60.0)
        self.match_threshold = self.get_parameter('match_threshold').value

        # BFMatcher with Hamming distance for ORB binary descriptors
        # crossCheck=False because we do the ratio test downstream in Node 5
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Store the previous frame's descriptors and keypoint positions
        self.prev_desc   = None
        self.prev_kp_x   = None
        self.prev_kp_y   = None

        self.subscription = self.create_subscription(
            DescriptorArray, '/descriptors', self.descriptor_callback, 10
        )
        self.publisher_ = self.create_publisher(MatchArray, '/raw_matches', 10)
        self.get_logger().info(
            f'FeatureMatchingNode started | threshold={self.match_threshold}'
        )

    def descriptor_callback(self, msg: DescriptorArray):
        n   = msg.num_descriptors
        sz  = msg.descriptor_size

        if n == 0:
            self.get_logger().warn('Received empty descriptor message.')
            return

        # Reconstruct the (N x descriptor_size) uint8 matrix from flat float list
        # ORB descriptors are uint8 but stored as float32 in the message
        curr_desc = np.array(msg.data, dtype=np.uint8).reshape((n, sz))
        curr_kp_x = list(msg.kp_x)
        curr_kp_y = list(msg.kp_y)

        if self.prev_desc is None:
            # First frame – store and wait for the next one
            self.prev_desc = curr_desc
            self.prev_kp_x = curr_kp_x
            self.prev_kp_y = curr_kp_y
            self.get_logger().info('First frame stored – waiting for next frame to match.')
            return

        # knnMatch returns the 2 best matches per descriptor (needed for ratio test in Node 5)
        raw_knn_matches = self.matcher.knnMatch(self.prev_desc, curr_desc, k=2)

        # Apply a loose distance threshold to remove very bad matches early
        query_x, query_y, train_x, train_y, distances = [], [], [], [], []

        for match_pair in raw_knn_matches:
            if len(match_pair) < 2:
                continue
            m, n_match = match_pair

            if m.distance < self.match_threshold:
                query_x.append(self.prev_kp_x[m.queryIdx])
                query_y.append(self.prev_kp_y[m.queryIdx])
                train_x.append(curr_kp_x[m.trainIdx])
                train_y.append(curr_kp_y[m.trainIdx])
                distances.append(float(m.distance))

        match_msg            = MatchArray()
        match_msg.header     = msg.header
        match_msg.query_x    = query_x
        match_msg.query_y    = query_y
        match_msg.train_x    = train_x
        match_msg.train_y    = train_y
        match_msg.distances  = distances
        match_msg.count      = len(query_x)

        self.publisher_.publish(match_msg)
        self.get_logger().debug(f'Published {match_msg.count} raw matches')

        # Roll the buffer: current frame becomes previous for the next iteration
        self.prev_desc  = curr_desc
        self.prev_kp_x  = curr_kp_x
        self.prev_kp_y  = curr_kp_y


def main(args=None):
    rclpy.init(args=args)
    node = FeatureMatchingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
