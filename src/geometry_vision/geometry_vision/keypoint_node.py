import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_interfaces.msg import KeypointArray


class KeypointDetectionNode(Node):

    def __init__(self):
        super().__init__('keypoint_detection_node')

        self.declare_parameter('max_keypoints', 500)
        max_kp = self.get_parameter('max_keypoints').value

        self.detector = cv2.ORB_create(nfeatures=max_kp)
        self.bridge   = CvBridge()

        self.subscription = self.create_subscription(
            Image, '/camera_frames', self.image_callback, 10
        )
        self.publisher_ = self.create_publisher(KeypointArray, '/keypoints', 10)
        self.get_logger().info(f'KeypointDetectionNode started | max_keypoints={max_kp}')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints = self.detector.detect(gray, None)

        if len(keypoints) < 20:
            self.get_logger().warn(
                f'Low keypoints detected: {len(keypoints)} (minimum required: 20)'
            )

        # VISUALIZATION: draw keypoints on the frame
        vis = cv2.drawKeypoints(
            frame, keypoints, None,
            color=(0, 255, 0),                          
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.putText(
            vis, f'Keypoints: {len(keypoints)}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )
        cv2.imshow('Keypoint Detection', vis)
        cv2.waitKey(1)

        kp_msg        = KeypointArray()
        kp_msg.header = msg.header
        kp_msg.x      = [kp.pt[0] for kp in keypoints]
        kp_msg.y      = [kp.pt[1] for kp in keypoints]
        kp_msg.count  = len(keypoints)

        self.publisher_.publish(kp_msg)
        self.get_logger().debug(f'Published {len(keypoints)} keypoints')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeypointDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
