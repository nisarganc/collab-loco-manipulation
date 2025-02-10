import rclpy
import numpy as np
from rclpy.node import Node

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from msgs_interfaces.msg import MarkerPoseArray, MarkerPose

MAX_LINEAR_VELOCITY = 0.001  # m/s
MAX_ANGULAR_VELOCITY = 2.84  # rad/s

IDs = {"/turtle2": 10, "/turtle4": 20, "/turtle6": 30, "object": 40}


class PosePIDController(Node):
    """
    Class for the Pose P Controller

    Attributes:
        kp_angular (float): Proportional gain for angular velocity
        kp_linear (float): Proportional gain for linear velocity
        subscription (Subscriber): Subscriber for the Aruco marker
        publisher (Publisher): Publisher for the cmd_vel
    """

    def __init__(self):
        """
        Constructor for the PosePController class

        Args:
            Node: Inherited from rclpy.Node
        """
        super().__init__("pose_p_controller")

        # PID parameters
        self.kp_linear = 0.2
        self.ki_linear = 0.0
        self.kd_linear = 0.0

        self.kp_angular = 0.2
        self.ki_angular = 0.0
        self.kd_angular = 0.0

        self.integral_linear = 0.0
        self.integral_angular = 0.0
        self.prev_error_linear = 0.0
        self.prev_error_angular = 0.0

        # Time step
        self.dt = 0.01  # 10ms

        # Robot and object pose
        self.robot_pose = MarkerPose()
        self.robot_pose.id = IDs[self.get_namespace()]
        self.desired_pose = MarkerPose()
        self.desired_pose.id = IDs["object"]

        # Create Subscriber for Aruco
        self.subscription = self.create_subscription(
            MarkerPoseArray, "/aruco_poses", self.pose_callback, 10
        )

        # Create Publisher for cmd_vel for the robot
        namespace = self.get_namespace()
        self.publisher = self.create_publisher(Twist, f"{namespace}/cmd_vel", 10)

        # Create timer
        self.timer = self.create_timer(self.dt, self.controller_callback)

    def pose_callback(self, msg: MarkerPoseArray):
        """
        Callback function for the subscriber

        Args:
            msg (Pose): Pose message from the Aruco marker
        """
        # Get the robot pose
        robot_pose = self.get_pose(msg, self.robot_pose.id)

        # Get the object pose
        object_pose = self.get_pose(msg, self.desired_pose.id)

        if None in robot_pose or None in object_pose:
            self.desired_pose.x, self.desired_pose.y, self.desired_pose.theta = (
                0.0,
                0.0,
                0.0,
            )
            return

        # Transform the robot pose to the object frame
        self.robot_pose.x, self.robot_pose.y, self.robot_pose.theta = (
            self.transform_to_frame(robot_pose, object_pose)
        )

        # Get the desired pose
        self.desired_pose.x, self.desired_pose.y, self.desired_pose.theta = (
            self.get_desired_pose(
                (self.robot_pose.x, self.robot_pose.y, self.robot_pose.theta)
            )
        )

    def controller_callback(self):
        """
        Callback function for the controller
        """
        if (
            self.desired_pose.x == 0
            and self.desired_pose.y == 0
            and self.desired_pose.theta == 0
        ):
            return

        # Calculate errors
        error_x = self.desired_pose.x - self.robot_pose.x
        error_y = self.desired_pose.y - self.robot_pose.y

        error_linear = np.sqrt(error_x**2 + error_y**2)
        error_angular = self.normalize_angle(
            self.desired_pose.theta - self.robot_pose.theta
        )

        # PID calculations
        self.integral_linear += error_linear * self.dt
        self.integral_angular += error_angular * self.dt

        derivative_linear = (error_linear - self.prev_error_linear) / self.dt
        derivative_angular = (error_angular - self.prev_error_angular) / self.dt

        v = (
            self.kp_linear * error_linear
            + self.ki_linear * self.integral_linear
            + self.kd_linear * derivative_linear
        )

        omega = (
            self.kp_angular * error_angular
            + self.ki_angular * self.integral_angular
            + self.kd_angular * derivative_angular
        )

        self.prev_error_angular = error_angular
        self.prev_error_linear = error_linear

        # Limit the velocity
        ratio = omega / v
        v = MAX_LINEAR_VELOCITY
        omega *= ratio

        # Publish the velocity
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.publisher.publish(cmd)

        # Print log
        self.get_logger().info(
            f"Robot Pose: ({self.robot_pose.x:.2f}, {self.robot_pose.y:.2f}, {self.robot_pose.theta:.2f}) "
            f"Desired Pose: ({self.desired_pose.x:.2f}, {self.desired_pose.y:.2f}, {self.desired_pose.theta:.2f}) "
            f"Delta: ({error_x:.2f}, {error_y:.2f}, {error_angular:.2f})"
        )

    def get_pose(self, msg: MarkerPoseArray, id: int) -> tuple:
        """
        Get the pose from the message

        Args:
            msg (Pose): Pose message from the Aruco marker
            id (int): ID of the robot

        Returns:
            tuple: x, y, theta
        """
        x, y, theta = None, None, None

        for pose in msg.poses:
            if pose.id == id:
                return pose.x, pose.y, pose.theta

        return x, y, theta

    def transform_to_frame(self, robot_pose: tuple, object_pose: tuple) -> tuple:
        """
        Transform the robot pose to the object frame

        Args:
            robot_pose (tuple): x, y, theta of the robot
            object_pose (tuple): x, y, theta of the object

        Returns:
            tuple: x, y, theta
        """
        x_robot, y_robot, theta_robot = robot_pose
        reference_x, reference_y, reference_angle = object_pose

        x = (x_robot - reference_x) * np.cos(reference_angle) + (
            y_robot - reference_y
        ) * np.sin(reference_angle)
        y = -(x_robot - reference_x) * np.sin(reference_angle) + (
            y_robot - reference_y
        ) * np.cos(reference_angle)
        theta = self.normalize_angle(theta_robot - reference_angle)

        return x, y, theta

    def get_desired_pose(self, robot_pose: tuple) -> tuple:
        """
        Get desired pose for the robot

        Args:
            robot_pose (tuple): x, y, theta of the robot

        Returns:
            tuple: x, y, theta
        """
        # Check at which edge of the object the robot is. The object is a square of 0.42m
        x, y, theta = robot_pose
        radius = 0.5

        if theta > -np.pi / 4 and theta < np.pi / 4:
            x_desired = -radius
            y_desired = 0.0
            theta_desired = 0
        elif theta > np.pi / 4 and theta < 3 * np.pi / 4:
            x_desired = 0.0
            y_desired = -radius
            theta_desired = np.pi / 2
        elif theta > 3 * np.pi / 4 or theta < -3 * np.pi / 4:
            x_desired = radius
            y_desired = 0.0
            theta_desired = np.pi
        else:
            x_desired = 0.0
            y_desired = radius
            theta_desired = -np.pi / 2

        x_desired = radius
        y_desired = 0.0
        theta_desired = np.pi

        return x_desired, y_desired, float(theta_desired)

    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle

        Args:
            angle (float): Angle to normalize

        Returns:
            float: Normalized angle
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = PosePIDController()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
