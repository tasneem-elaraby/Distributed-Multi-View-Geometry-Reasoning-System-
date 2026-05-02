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

        raw = msg.distances

        # Guard: if distances are NOT interleaved (legacy / unexpected format),
        # fall back to the old percentile approach so the node stays robust.
        if len(raw) == msg.count * 2:
            d1_list = raw[0::2]   # best distances
            d2_list = raw[1::2]   # second-best distances
            use_true_ratio = True
        else:
            self.get_logger().warn(
                'distances field length does not match expected interleaved format. '
                'Falling back to percentile-based filtering.'
            )
            d1_list = raw
            d2_list = None
            use_true_ratio = False

        if use_true_ratio:
            good_indices = []
            for i, (d1, d2) in enumerate(zip(d1_list, d2_list)):
                if d2 == 0.0:
                    # Avoid division by zero; keep only if d1 is also zero (perfect match)
                    if d1 == 0.0:
                        good_indices.append(i)
                    continue
                if d1 / d2 < self.ratio_thresh:
                    good_indices.append(i)
        else:
            # Fallback: keep the best ratio_thresh fraction by distance
            sorted_dist  = sorted(enumerate(d1_list), key=lambda x: x[1])
            cutoff       = max(1, int(len(sorted_dist) * self.ratio_thresh))
            good_indices = [idx for idx, _ in sorted_dist[:cutoff]]

        if len(good_indices) == 0:
            self.get_logger().warn('All matches rejected by ratio test.')
            self._publish_empty(msg)
            return

        best_for_train = {}
        for i in good_indices:
            train_key = (round(msg.train_x[i], 1), round(msg.train_y[i], 1))
            d1 = d1_list[i]
            if train_key not in best_for_train or d1 < best_for_train[train_key][1]:
                best_for_train[train_key] = (i, d1)

        filtered_indices = [v[0] for v in best_for_train.values()]

        # Also enforce uniqueness of query (previous-frame) points
        best_for_query = {}
        for i in filtered_indices:
            query_key = (round(msg.query_x[i], 1), round(msg.query_y[i], 1))
            d1 = d1_list[i]
            if query_key not in best_for_query or d1 < best_for_query[query_key][1]:
                best_for_query[query_key] = (i, d1)

        filtered_indices = [v[0] for v in best_for_query.values()]

    
        filtered_msg           = MatchArray()
        filtered_msg.header    = msg.header
        filtered_msg.query_x   = [msg.query_x[i]  for i in filtered_indices]
        filtered_msg.query_y   = [msg.query_y[i]  for i in filtered_indices]
        filtered_msg.train_x   = [msg.train_x[i]  for i in filtered_indices]
        filtered_msg.train_y   = [msg.train_y[i]  for i in filtered_indices]
        filtered_msg.distances = [float(d1_list[i]) for i in filtered_indices]
        filtered_msg.count     = len(filtered_indices)

        self.publisher_.publish(filtered_msg)
        self.get_logger().info(
            f'Filtering: {msg.count} raw -> {filtered_msg.count} filtered '
            f'(ratio_thresh={self.ratio_thresh})'
        )


    def _publish_empty(self, original_msg):
        empty        = MatchArray()
        empty.header = original_msg.header
        empty.count  = 0
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
