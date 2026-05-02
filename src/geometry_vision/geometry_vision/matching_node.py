import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from geometry_interfaces.msg import DescriptorArray, MatchArray


class FeatureMatchingNode(Node):

    def __init__(self):
        super().__init__('feature_matching_node')

        self.declare_parameter('match_threshold', 100.0)
        self.match_threshold = self.get_parameter('match_threshold').value

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_desc  = None
        self.prev_kp_x  = None
        self.prev_kp_y  = None
        self.prev_frame = None

        self.subscription = self.create_subscription(
            DescriptorArray, '/descriptors', self.descriptor_callback, 10
        )
        self.publisher_ = self.create_publisher(MatchArray, '/raw_matches', 10)
        self.get_logger().info(
            f'FeatureMatchingNode started | threshold={self.match_threshold}'
        )

    def descriptor_callback(self, msg: DescriptorArray):
        num = msg.num_descriptors
        sz  = msg.descriptor_size

        if num == 0:
            self.get_logger().warn('Received empty descriptor message.')
            return

        curr_desc = np.array(msg.data, dtype=np.uint8).reshape((num, sz))
        curr_kp_x = list(msg.kp_x)
        curr_kp_y = list(msg.kp_y)

        # Build a simple black frame with keypoint dots for visualization
        h, w = 480, 640
        curr_frame = np.zeros((h, w, 3), dtype=np.uint8)
        for x, y in zip(curr_kp_x, curr_kp_y):
            cv2.circle(curr_frame, (int(x), int(y)), 3, (0, 200, 255), -1)

        if self.prev_desc is None:
            self.prev_desc  = curr_desc
            self.prev_kp_x  = curr_kp_x
            self.prev_kp_y  = curr_kp_y
            self.prev_frame = curr_frame
            self.get_logger().info('First frame stored – waiting for next frame to match.')
            return

        raw_knn_matches = self.matcher.knnMatch(self.prev_desc, curr_desc, k=2)

        query_x    = []
        query_y    = []
        train_x    = []
        train_y    = []
        distances  = []
        distances2 = []

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
                distances2.append(float(n_match.distance))

        # VISUALIZATION: side-by-side prev | curr with match lines
        vis = np.hstack([self.prev_frame, curr_frame])
        for qx, qy, tx, ty in zip(query_x, query_y, train_x, train_y):
            pt1 = (int(qx), int(qy))
            pt2 = (int(tx) + w, int(ty))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(
            vis, f'Raw Matches: {len(query_x)}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        cv2.imshow('Feature Matching  [prev | curr]', vis)
        cv2.waitKey(1)

        interleaved = []
        for d1, d2 in zip(distances, distances2):
            interleaved.append(d1)
            interleaved.append(d2)

        match_msg           = MatchArray()
        match_msg.header    = msg.header
        match_msg.query_x   = query_x
        match_msg.query_y   = query_y
        match_msg.train_x   = train_x
        match_msg.train_y   = train_y
        match_msg.distances = interleaved
        match_msg.count     = len(query_x)

        self.publisher_.publish(match_msg)
        self.get_logger().debug(f'Published {match_msg.count} raw matches')

        self.prev_desc  = curr_desc
        self.prev_kp_x  = curr_kp_x
        self.prev_kp_y  = curr_kp_y
        self.prev_frame = curr_frame

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


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
