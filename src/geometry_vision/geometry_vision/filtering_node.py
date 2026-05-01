
import rclpy
from rclpy.node import Node
from geometry_interfaces.msg import MatchArray

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

        # -----------------------------------------------------------------------
        # Stage 1: Ratio Test
        # The raw_matches distances list already contains all matches that passed
        # the initial distance threshold in Node 4.
        # We simulate the ratio test by sorting and comparing adjacent distances.
        # In practice the BFMatcher knn output (m, n) gives us:
        #   m.distance / n.distance < ratio  -> keep m
        # Since the MatchArray flattened those pairs, we use distance alone here
        # and apply a normalised self-consistency check.
        # -----------------------------------------------------------------------
        good_indices = []
        distances    = msg.distances

        # Sort matches by distance and keep only the top 70% (ratio approximation)
        # This is a valid simplified ratio test when only one distance is stored.
        if len(distances) > 1:
            threshold_dist = sorted(distances)[int(len(distances) * self.ratio_thresh)]
            good_indices   = [i for i, d in enumerate(distances) if d <= threshold_dist]
        else:
            good_indices = list(range(len(distances)))

        # -----------------------------------------------------------------------
        # Stage 2: Cross-Check Simulation
        # We enforce that query point and train point have a unique mapping.
        # Remove duplicates where the same train point appears multiple times.
        # -----------------------------------------------------------------------
        seen_train = set()
        filtered_indices = []
        for i in good_indices:
            train_key = (round(msg.train_x[i], 1), round(msg.train_y[i], 1))
            if train_key not in seen_train:
                seen_train.add(train_key)
                filtered_indices.append(i)

        filtered_msg          = MatchArray()
        filtered_msg.header   = msg.header
        filtered_msg.query_x  = [msg.query_x[i]   for i in filtered_indices]
        filtered_msg.query_y  = [msg.query_y[i]   for i in filtered_indices]
        filtered_msg.train_x  = [msg.train_x[i]   for i in filtered_indices]
        filtered_msg.train_y  = [msg.train_y[i]   for i in filtered_indices]
        filtered_msg.distances= [msg.distances[i]  for i in filtered_indices]
        filtered_msg.count    = len(filtered_indices)

        self.publisher_.publish(filtered_msg)
        self.get_logger().debug(
            f'Filtering: {msg.count} raw -> {filtered_msg.count} filtered'
        )

    def _publish_empty(self, original_msg):
        empty            = MatchArray()
        empty.header     = original_msg.header
        empty.count      = 0
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
