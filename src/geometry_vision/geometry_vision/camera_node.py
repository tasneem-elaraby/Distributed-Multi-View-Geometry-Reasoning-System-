
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraStreamNode(Node):

    def __init__(self):
        super().__init__('camera_stream_node')

        # Declare parameters with defaults
        self.declare_parameter('camera_source', 0)
        self.declare_parameter('frame_rate', 10.0)

        camera_source = self.get_parameter('camera_source').value
        frame_rate    = self.get_parameter('frame_rate').value

        # Enforce minimum frame rate as per system rules
        if frame_rate < 5.0:
            self.get_logger().warn('frame_rate below minimum (5 FPS). Forcing 5 FPS.')
            frame_rate = 5.0

        # Try to interpret camera_source as integer (webcam index) or string (file path)
        try:
            source = int(camera_source)
        except (ValueError, TypeError):
            source = str(camera_source)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera source: {source}')
            raise RuntimeError('Camera source could not be opened.')

        self.bridge     = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/camera_frames', 10)

        # Timer drives the capture loop at the requested frame rate
        self.timer = self.create_timer(1.0 / frame_rate, self.capture_and_publish)
        self.get_logger().info(f'CameraStreamNode started | source={source} | rate={frame_rate} Hz')

    def capture_and_publish(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to read frame – end of stream or camera error.')
            return

        # Convert OpenCV BGR image to ROS2 Image message
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ros_image.header.stamp    = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera'

        self.publisher_.publish(ros_image)

    def destroy_node(self):
        # Release the video capture resource cleanly
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
