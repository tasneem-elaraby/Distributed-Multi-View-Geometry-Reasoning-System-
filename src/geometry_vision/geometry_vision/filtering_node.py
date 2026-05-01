import rclpy
from rclpy.node import Node
from surveillance_interfaces.msg import MatchArray


class MatchFilteringNode(Node):

    def __init__(self):
        super().__init__('match_filtering_node')

        self.declare_parameter('ratio_test_threshold', 0.75)
        self.ratio_thresh = self.get_parameter('ratio_test_threshold').value

        self.subscription = self.create_subscription(
            MatchArray, '/raw_matches', self.match_callback, 10
        )
        self.publisher_ = self.create_publisher(MatchArray, '/filtered_matches', 10)

        self.get_logger().info(
            f'MatchFilteringNode started | ratio_threshold={self.ratio_thresh}'
        )

    def match_callback(self, msg: MatchArray):

        if msg.count == 0:
            self.get_logger().warn('Received empty match list.')
            self._publish_empty(msg)
            return

        # Ensure we have both m and n distances
        if not hasattr(msg, 'm_distances') or not hasattr(msg, 'n_distances'):
            self.get_logger().error('MatchArray missing m_distances / n_distances → cannot apply ratio test!')
            return
      
        good_indices = []

        # keep match if: m.distance / n.distance < threshold
        for i in range(msg.count):
            m_dist = msg.m_distances[i]
            n_dist = msg.n_distances[i]

            if n_dist == 0:
                continue  # avoid division by zero

            ratio = m_dist / n_dist

            if ratio < self.ratio_thresh:
                good_indices.append(i)

        if len(good_indices) == 0:
            self.get_logger().warn('All matches rejected by ratio test.')
            self._publish_empty(msg)
            return
        
        #  Cross-Check (one-to-one mapping)
        # Ensure each train keypoint appears only once
        seen_train = set()
        filtered_indices = []

        for i in good_indices:
            train_key = (round(msg.train_x[i], 1), round(msg.train_y[i], 1))

            if train_key not in seen_train:
                seen_train.add(train_key)
                filtered_indices.append(i)
        # Build filtered message
        
        filtered_msg = MatchArray()
        filtered_msg.header = msg.header
        filtered_msg.query_x = [msg.query_x[i] for i in filtered_indices]
        filtered_msg.query_y = [msg.query_y[i] for i in filtered_indices]
        filtered_msg.train_x = [msg.train_x[i] for i in filtered_indices]
        filtered_msg.train_y = [msg.train_y[i] for i in filtered_indices]

        # Keep ONLY m distances as final match score
        filtered_msg.distances = [msg.m_distances[i] for i in filtered_indices]
        filtered_msg.count = len(filtered_indices)

        self.publisher_.publish(filtered_msg)

        self.get_logger().info(
            f'Filtering: {msg.count} raw → {filtered_msg.count} filtered'
        )

    def _publish_empty(self, original_msg):
        empty = MatchArray()
        empty.header = original_msg.header
        empty.count = 0
        self.publisher_.publish(empty)


def main(args=None):
    rclpy.init(args=args)
    node = MatchFilteringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
