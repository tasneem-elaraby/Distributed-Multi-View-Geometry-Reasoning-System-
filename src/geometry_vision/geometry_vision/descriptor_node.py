import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_interfaces.msg import KeypointArray, DescriptorArray


class DescriptorExtractionNode(Node):

    def __init__(self):
        super().__init__('descriptor_extraction_node')

        self.declare_parameter('descriptor_type', 'ORB')
        desc_type = self.get_parameter('descriptor_type').value.upper()

       
        if desc_type == 'SIFT':
            self.extractor = cv2.SIFT_create()
            self.norm_type = cv2.NORM_L2
        elif desc_type == 'BRISK':
            self.extractor = cv2.BRISK_create()
            self.norm_type = cv2.NORM_HAMMING
        else:
            if desc_type != 'ORB':
                self.get_logger().warn(
                    f"Unknown descriptor_type '{desc_type}'. Falling back to ORB."
                )
            desc_type      = 'ORB'
            self.extractor = cv2.ORB_create()
            self.norm_type = cv2.NORM_HAMMING

        self.bridge = CvBridge()

        # Cache for the latest frame and keypoint message
        self.latest_frame     = None
        self.latest_keypoints = None

        self.sub_image = self.create_subscription(
            Image, '/camera_frames', self.image_callback, 10
        )
        self.sub_kp = self.create_subscription(
            KeypointArray, '/keypoints', self.keypoint_callback, 10
        )
        self.publisher_ = self.create_publisher(DescriptorArray, '/descriptors', 10)
        self.get_logger().info(
            f'DescriptorExtractionNode started | type={desc_type}'
        )

    def image_callback(self, msg: Image):
        self.latest_frame = msg
        self._try_compute()

    def keypoint_callback(self, msg: KeypointArray):
        self.latest_keypoints = msg
        self._try_compute()

    def _try_compute(self):
        """Compute descriptors only when both frame and keypoints are available."""
        if self.latest_frame is None or self.latest_keypoints is None:
            return

        frame = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding='bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rebuild cv2.KeyPoint objects from the published float arrays
        kp_list = [
            cv2.KeyPoint(x=x, y=y, size=10.0)
            for x, y in zip(self.latest_keypoints.x, self.latest_keypoints.y)
        ]

        if len(kp_list) == 0:
            self.get_logger().warn('No keypoints to compute descriptors for.')
            return

        # compute() fills in the descriptor for each KeyPoint
        keypoints, descriptors = self.extractor.compute(gray, kp_list)

        if descriptors is None or len(descriptors) == 0:
            self.get_logger().warn('Descriptor computation returned empty result.')
            return

        descriptors_uint8 = np.clip(descriptors, 0, 255).astype(np.uint8)

        # Flatten the descriptor matrix (N x D) into a 1D list for the message
        desc_msg                 = DescriptorArray()
        desc_msg.header          = self.latest_keypoints.header
        desc_msg.data            = descriptors_uint8.flatten().tolist()
        desc_msg.num_descriptors = int(descriptors_uint8.shape[0])
        desc_msg.descriptor_size = int(descriptors_uint8.shape[1])
        desc_msg.kp_x            = [kp.pt[0] for kp in keypoints]
        desc_msg.kp_y            = [kp.pt[1] for kp in keypoints]

        self.publisher_.publish(desc_msg)
        self.get_logger().debug(
            f'Published {desc_msg.num_descriptors} descriptors '
            f'(size={desc_msg.descriptor_size})'
        )

        # Reset cache so we don't reprocess the same pair
        self.latest_frame     = None
        self.latest_keypoints = None

def main(args=None):
    rclpy.init(args=args)
    node = DescriptorExtractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
