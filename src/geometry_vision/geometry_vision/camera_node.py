#!/usr/bin/env python3

import rclpy                #rospy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraStreamNode(Node):       # rospy.init_node()
    def __init__(self):
        super().__init__('camera_stream_node')

        # Parameters
        self.declare_parameter('camera_source', "0")   # 0 = webcam or video path #rospy.get_param()
        self.declare_parameter('frame_rate', 10)

        self.camera_source = self.get_parameter('camera_source').value
        self.frame_rate = self.get_parameter('frame_rate').value

        if self.frame_rate <= 0:
            self.get_logger().warn("Invalid frame_rate, using default 10 FPS")          #rospy.logwarn()
            self.frame_rate = 10

        # Publisher
        self.publisher_ = self.create_publisher(Image, '/camera_frames', 10)

        # Open camera/video
        self.cap = self.open_source(self.camera_source)

        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error(f'Cannot open source: {self.camera_source}')
            self.cap = None
            return

        self.bridge = CvBridge()

        period = 1.0 / self.frame_rate                          #rate = rospy.Rate(10)  while not rospy.is_shutdown():
        self.timer = self.create_timer(period, self.publish_frame)

        self.get_logger().info(
            f'Camera node started | source={self.camera_source} | fps={self.frame_rate}'
        )

    def open_source(self, source):
        if str(source).isdigit():
            return cv2.VideoCapture(int(source))
        else:
            return cv2.VideoCapture(source)

    def publish_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Restarting video...")

            # reset to first frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # try again
            ret, frame = self.cap.read()

            # if still failed → reopen
            if not ret:
                self.get_logger().warn("Reopening video source...")
                self.cap.release()
                self.cap = self.open_source(self.camera_source)
                return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()     #rospy.Time.now()

        self.publisher_.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
