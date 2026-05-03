

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from std_msgs.msg import String

from geometry_interfaces.msg import CameraMotion, GeometricInliers
from geometry_interfaces.action import ReportAction


class ReliabilityDecisionNode(Node):

    def __init__(self):
        super().__init__('reliability_decision_node')

        self.declare_parameter('min_inliers', 10)
        self.min_inliers = self.get_parameter('min_inliers').value

        # Cache for the latest messages from both upstream nodes
        self.latest_motion  = None
        self.latest_inliers = None

        # Subscriptions
        self.sub_motion = self.create_subscription(
            CameraMotion, '/camera_motion', self.motion_callback, 10
        )
        self.sub_inliers = self.create_subscription(
            GeometricInliers, '/geometric_inliers', self.inlier_callback, 10
        )

        # Publisher for human-readable system state
        self.state_publisher = self.create_publisher(String, '/system_state', 10)

        # Action server for external report requests
        self._action_server = ActionServer(
            self,
            ReportAction,
            '/report_action',
            execute_callback=self.execute_report_action,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info(
            f'ReliabilityDecisionNode started | min_inliers={self.min_inliers}'
        )

    # ------------------------------------------------------------------
    # Topic callbacks
    # ------------------------------------------------------------------
    def motion_callback(self, msg: CameraMotion):
        self.latest_motion = msg
        self._evaluate_and_publish()

    def inlier_callback(self, msg: GeometricInliers):
        self.latest_inliers = msg
        self._evaluate_and_publish()

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------
    def _evaluate_and_publish(self):
        """Decide system state based on the latest available data."""
        if self.latest_inliers is None:
            return   # Not enough data yet

        state = self._compute_state()

        state_msg      = String()
        state_msg.data = state
        self.state_publisher.publish(state_msg)
        self.get_logger().info(f'System State → {state}')

    def _compute_state(self) -> str:
        """
        Reliability rules:
        1. LOW_FEATURES : inlier_count == 0 (pipeline starved)
        2. UNRELIABLE   : inlier_count < min_inliers OR inlier_ratio < 0.3
        3. RELIABLE     : inlier_count >= min_inliers AND inlier_ratio >= 0.3
        """
        inlier_count = self.latest_inliers.inlier_count
        inlier_ratio = self.latest_inliers.inlier_ratio

        if inlier_count == 0:
            return 'LOW_FEATURES'

        if inlier_count < self.min_inliers or inlier_ratio < 0.3:
            return 'UNRELIABLE'

        return 'RELIABLE'

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------
    def goal_callback(self, goal_request):
        self.get_logger().info('ReportAction: Goal received')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('ReportAction: Cancel received')
        return CancelResponse.ACCEPT

    async def execute_report_action(self, goal_handle):
        """
        Action execution:
        1. Send feedback with current status
        2. Compute final state
        3. Return result
        """
        self.get_logger().info('ReportAction: Executing report...')

        # Send intermediate feedback
        feedback_msg               = ReportAction.Feedback()
        feedback_msg.current_status = 'Evaluating system state...'
        goal_handle.publish_feedback(feedback_msg)

        # Compute the state
        state = self._compute_state() if self.latest_inliers else 'LOW_FEATURES'

        inlier_count = self.latest_inliers.inlier_count if self.latest_inliers else 0
        inlier_ratio = self.latest_inliers.inlier_ratio if self.latest_inliers else 0.0

        # Mark goal as succeeded and return result
        goal_handle.succeed()

        result              = ReportAction.Result()
        result.accepted     = True
        result.message      = (
            f'State={state} | inliers={inlier_count} | ratio={inlier_ratio:.2f}'
        )
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ReliabilityDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
