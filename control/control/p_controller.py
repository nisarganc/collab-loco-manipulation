import rclpy
import numpy as np
from rclpy.node import Node

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from msgs_interfaces.msg import MarkerPoseArray

MAX_LINEAR_VELOCITY = 0.1  # m/s
MAX_ANGULAR_VELOCITY = 2.84  # rad/s

IDs = {"turtle2": 10, "turtle4": 20, "turtle6": 30, "object": 40}


class PosePController(Node):
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

        # Controller gains
        self.kp_angular = 0.2
        self.kp_linear = 0.2

        # Create Subscriber for Aruco
        self.subscription = self.create_subscription(
            MarkerPoseArray, "aruco_poses", self.pose_callback, 10
        )

        # Create Publisher for cmd_vel
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def pose_callback(self, msg: MarkerPoseArray):
        """
        Callback function for the subscriber

        Args:
            msg (Pose): Pose message from the Aruco marker
        """
        # Initialize variables
        x, y, theta = None, None, None

        # Get current pose from the robot
        robot_pose = self.get_pose(msg, IDs[self.get_namespace()])
        object_pose = self.get_pose(msg, IDs["object"])

        # Get the robot pose in the object frame
        if None in robot_pose or None in object_pose:
            return
        else:
            robot_pose_in_obj_frame = self.transform_coordinates_to_object_frame(
                robot_pose, object_pose
            )
            x, y, theta = robot_pose_in_obj_frame

        # Get desired pose
        x_desired, y_desired, theta_desired = self.get_desired_pose(robot_pose_in_obj_frame)

        # Linear velocity
        error_x = x_desired - x
        error_y = y_desired - y
        linear_vel = self.kp_linear * np.sqrt(error_x**2 + error_y**2)

        # Angular velocity
        error_theta = self.normalize_angle(theta_desired - theta)
        angular_vel = self.kp_angular * error_theta

        # Normalize the velocities
        linear_angular_ratio = linear_vel / angular_vel
        if linear_vel > MAX_LINEAR_VELOCITY:
            linear_vel = MAX_LINEAR_VELOCITY
            angular_vel = linear_vel / linear_angular_ratio

        # Check if threshold is reached (optional)
        if (
            np.abs(error_x) < 0.05
            and np.abs(error_y) < 0.05
            and np.abs(error_theta) < 0.05
        ):
            linear_vel = 0.0
            angular_vel = 0.0

        # Publish the velocity
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.publisher.publish(cmd_vel_msg)

        # Print log
        self.get_logger().info(
            f"Robot Pose: ({x:.2f}, {y:.2f}, {theta:.2f}) "
            f"Desired Pose: ({x_desired:.2f}, {y_desired:.2f}, {theta_desired:.2f}) "
            f"Delta: ({error_x:.2f}, {error_y:.2f}, {error_theta:.2f})"
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

    def transform_coordinates_to_object_frame(
        self, robot_pose: Pose, object_pose: Pose
    ) -> tuple:
        """
        Transform the robot pose to the object frame

        Args:
            robot_pose (tuple): x, y, theta of the robot
            object_pose (tuple): x, y, theta of the object

        Returns:
            tuple: x, y, theta
        """
        x_robot, y_robot, theta_robot = robot_pose
        x_object, y_object, theta_object = object_pose

        x = (x_robot - x_object) * np.cos(theta_object) + (
            y_robot - y_object
        ) * np.sin(theta_object)
        y = -(x_robot - x_object) * np.sin(theta_object) + (
            y_robot - y_object
        ) * np.cos(theta_object)
        theta = self.normalize_angle(theta_robot - theta_object)

        return x, y, theta

    def get_desired_pose(self, robot_pose: Pose) -> tuple:
        """
        Get desired pose for the robot

        Args:
            robot_pose (tuple): x, y, theta of the robot in the object frame

        Returns:
            tuple: x, y, theta
        """
        # Check at which edge of the object the robot is. The object is a square of 0.42m
        x, y, theta = robot_pose
        
        if theta > -np.pi/4 and theta < np.pi/4:
            x_desired = -0.2
            y_desired = 0
            theta_desired = 0

        elif theta > np.pi/4 and theta < 3*np.pi/4:
            x_desired = 0
            y_desired = 0.2
            theta_desired = np.pi/2

        elif theta > 3*np.pi/4 or theta < -3*np.pi/4:
            x_desired = 0.2
            y_desired = 0
            theta_desired = np.pi

        else:
            x_desired = 0
            y_desired = -0.2
            theta_desired = -np.pi/2

        return x_desired, y_desired, theta_desired

    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle

        Args:
            angle (float): Angle to normalize

        Returns:
            float: Normalized angle
        """
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        return angle


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = PosePController()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
