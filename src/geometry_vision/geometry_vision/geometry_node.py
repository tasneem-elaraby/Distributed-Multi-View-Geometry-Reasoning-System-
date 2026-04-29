
import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from surveillance_interfaces.msg import MatchArray, GeometricInliers
from surveillance_interfaces.srv import CheckGeometry


class GeometricConsistencyNode(Node):

    def __init__(self):
        super().__init__('geometric_consistency_node')

        self.declare_parameter('inlier_threshold', 3.0)
        self.inlier_thresh = self.get_parameter('inlier_threshold').value

        self.subscription = self.create_subscription(
            MatchArray, '/filtered_matches', self.match_callback, 10
        )
        self.publisher_ = self.create_publisher(
            GeometricInliers, '/geometric_inliers', 10
        )

        # Service for on-demand geometric validation
        self.srv = self.create_service(
            CheckGeometry, '/check_geometry', self.check_geometry_callback
        )

        self.get_logger().info(
            f'GeometricConsistencyNode started | inlier_threshold={self.inlier_thresh} px'
        )

    # ------------------------------------------------------------------
    # Core geometry check (used by both topic and service paths)
    # ------------------------------------------------------------------
    def _compute_inliers(self, qx, qy, tx, ty):
        """
        Estimate the Fundamental Matrix using RANSAC and return inlier mask.

        The Fundamental Matrix F encodes the epipolar constraint:
            p2^T * F * p1 = 0
        for a point p1 in frame1 and its corresponding point p2 in frame2.

        Returns (inlier_mask, F) where inlier_mask is a boolean array.
        """
        pts1 = np.array(list(zip(qx, qy)), dtype=np.float32)
        pts2 = np.array(list(zip(tx, ty)), dtype=np.float32)

        if len(pts1) < 8:
            # Fundamental matrix requires at least 8 point correspondences
            return None, None

        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.inlier_thresh,
            confidence=0.99
        )

        return mask, F

    # ------------------------------------------------------------------
    # Topic subscription handler
    # ------------------------------------------------------------------
    def match_callback(self, msg: MatchArray):
        if msg.count < 8:
            self.get_logger().warn(
                f'Too few matches ({msg.count}) for Fundamental Matrix estimation. '
                f'Need at least 8.'
            )
            self._publish_empty(msg)
            return

        mask, F = self._compute_inliers(
            msg.query_x, msg.query_y,
            msg.train_x, msg.train_y
        )

        if mask is None:
            self._publish_empty(msg)
            return

        mask_flat = mask.ravel().astype(bool)

        inlier_qx = [msg.query_x[i] for i, v in enumerate(mask_flat) if v]
        inlier_qy = [msg.query_y[i] for i, v in enumerate(mask_flat) if v]
        inlier_tx = [msg.train_x[i] for i, v in enumerate(mask_flat) if v]
        inlier_ty = [msg.train_y[i] for i, v in enumerate(mask_flat) if v]

        inlier_count  = int(np.sum(mask_flat))
        outlier_count = msg.count - inlier_count
        inlier_ratio  = inlier_count / msg.count if msg.count > 0 else 0.0

        out_msg                  = GeometricInliers()
        out_msg.header           = msg.header
        out_msg.inlier_query_x   = inlier_qx
        out_msg.inlier_query_y   = inlier_qy
        out_msg.inlier_train_x   = inlier_tx
        out_msg.inlier_train_y   = inlier_ty
        out_msg.inlier_count     = inlier_count
        out_msg.outlier_count    = outlier_count
        out_msg.inlier_ratio     = float(inlier_ratio)

        self.publisher_.publish(out_msg)
        self.get_logger().debug(
            f'Geometry: {inlier_count} inliers / {msg.count} matches '
            f'(ratio={inlier_ratio:.2f})'
        )

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------
    def check_geometry_callback(self, request, response):
        mask, _ = self._compute_inliers(
            request.query_x, request.query_y,
            request.train_x, request.train_y
        )

        if mask is None:
            response.inlier_count  = 0
            response.inlier_ratio  = 0.0
            response.is_consistent = False
            return response

        inlier_count = int(np.sum(mask.ravel().astype(bool)))
        total        = request.count if request.count > 0 else len(request.query_x)

        response.inlier_count  = inlier_count
        response.inlier_ratio  = float(inlier_count / total) if total > 0 else 0.0
        response.is_consistent = response.inlier_ratio > 0.5

        return response

    def _publish_empty(self, original_msg):
        empty               = GeometricInliers()
        empty.header        = original_msg.header
        empty.inlier_count  = 0
        empty.outlier_count = original_msg.count
        empty.inlier_ratio  = 0.0
        self.publisher_.publish(empty)


def main(args=None):
    rclpy.init(args=args)
    node = GeometricConsistencyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
