

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from surveillance_interfaces.msg import GeometricInliers, CameraMotion


class MotionEstimationNode(Node):

    def __init__(self):
        super().__init__('motion_estimation_node')

        self.declare_parameter('focal_length', 700.0)
        self.focal = self.get_parameter('focal_length').value

        self.subscription = self.create_subscription(
            GeometricInliers, '/geometric_inliers', self.inlier_callback, 10
        )
        self.publisher_ = self.create_publisher(CameraMotion, '/camera_motion', 10)
        self.get_logger().info(
            f'MotionEstimationNode started | focal_length={self.focal} px'
        )

    def inlier_callback(self, msg: GeometricInliers):
        motion_msg                    = CameraMotion()
        motion_msg.header             = msg.header
        motion_msg.horizontal         = 'NONE'
        motion_msg.depth              = 'NONE'
        motion_msg.translation_magnitude = 0.0
        motion_msg.rotation_magnitude    = 0.0
        motion_msg.scale_ambiguity    = True   # always true for monocular

        if msg.inlier_count < 8:
            self.get_logger().warn(
                f'Not enough inliers ({msg.inlier_count}) for motion estimation.'
            )
            self.publisher_.publish(motion_msg)
            return

        pts1 = np.array(
            list(zip(msg.inlier_query_x, msg.inlier_query_y)), dtype=np.float32
        )
        pts2 = np.array(
            list(zip(msg.inlier_train_x, msg.inlier_train_y)), dtype=np.float32
        )

        # Assume principal point at (640, 360) for a typical 1280x720 stream
        cx, cy = 640.0, 360.0
        K = np.array([
            [self.focal, 0,          cx],
            [0,          self.focal, cy],
            [0,          0,          1 ]
        ], dtype=np.float64)

        # Compute the Fundamental Matrix from inliers, then derive Essential Matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

        if F is None or F.shape != (3, 3):
            self.get_logger().warn('Fundamental Matrix estimation failed.')
            self.publisher_.publish(motion_msg)
            return

        # E = K^T * F * K
        E = K.T @ F @ K

        # Decompose E into rotation R and translation t
        # recoverPose returns the rotation and unit translation vector
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        # ------------------------------------------------------------
        # Interpret translation direction
        # t is a unit vector [tx, ty, tz] in camera coordinates:
        #   tx < 0 → camera moved LEFT    tx > 0 → RIGHT
        #   tz > 0 → camera moved FORWARD  tz < 0 → BACKWARD
        # ------------------------------------------------------------
        tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        DIRECTION_THRESH = 0.1  # ignore tiny movements

        if abs(tx) > DIRECTION_THRESH:
            motion_msg.horizontal = 'LEFT' if tx < 0 else 'RIGHT'

        if abs(tz) > DIRECTION_THRESH:
            motion_msg.depth = 'FORWARD' if tz > 0 else 'BACKWARD'

        # Rotation magnitude: angle of the rotation vector (Rodrigues)
        rvec, _ = cv2.Rodrigues(R)
        motion_msg.rotation_magnitude    = float(np.linalg.norm(rvec))
        motion_msg.translation_magnitude = float(np.linalg.norm(t))
        # Scale is always ambiguous for monocular vision (|t| = 1 always)
        motion_msg.scale_ambiguity = True

        self.publisher_.publish(motion_msg)
        self.get_logger().info(
            f'Motion -> horizontal={motion_msg.horizontal} | '
            f'depth={motion_msg.depth} | '
            f'rot={motion_msg.rotation_magnitude:.3f} rad'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MotionEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
