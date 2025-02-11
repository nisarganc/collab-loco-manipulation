import rclpy
import numpy as np
from rclpy.node import Node
import threading

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from msgs_interfaces.msg import MarkerPoseArray, MarkerPose

MAX_LINEAR_VELOCITY = 0.1  # m/s
MAX_ANGULAR_VELOCITY = 2.84  # rad/s

IDs = {"/turtle2": 10, "/turtle4": 20, "/turtle6": 30, "object": 40}


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
        self._lock = threading.Lock()

        # P parameters
        self.kp_linear = 0.4
        self.kp_angular = 0.1
        self.kp_final_angle = 0.4

        # Time step
        self.dt = 0.1  # 100ms

        # Velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Create Subscriber for Aruco
        self.subscription = self.create_subscription(
            MarkerPoseArray, "/aruco_poses", self.pose_callback, 10
        )

        # Create Publisher for cmd_vel for the robot
        namespace = self.get_namespace().strip("/")
        if namespace:
            topic_name = f"/{namespace}/cmd_vel"
        else:
            topic_name = "cmd_vel"
        self.publisher = self.create_publisher(Twist, topic_name, 10)

        # Create timer
        self.timer = self.create_timer(self.dt, self.controller_callback)
        # self.timer_backwards = self.create_timer(1, self.backwards_callback)
        self.pose_callback_counter = 0

    def pose_callback(self, msg: MarkerPoseArray):
        """
        Callback function for the subscriber

        Args:
            msg (Pose): Pose message from the Aruco marker
        """
        # Get the poses
        robot_pose = self.get_pose(msg, IDs[self.get_namespace()])
        object_pose = self.get_pose(msg, IDs["object"])
        if None in robot_pose or None in object_pose:
            return

        desired_pose = self.get_desired_pose(robot_pose, object_pose)

        robot_pose_x, robot_pose_y, robot_pose_theta = robot_pose
        desired_pose_x, desired_pose_y, desired_pose_theta = desired_pose

        # Calculate errors
        error_x = desired_pose_x - robot_pose_x
        error_y = desired_pose_y - robot_pose_y

        error_linear = np.sqrt(error_x**2 + error_y**2)
        error_angular = self.normalize_angle(np.arctan2(error_y, error_x))

        # Final angle
        error_final_angle = self.normalize_angle(desired_pose_theta - robot_pose_theta)

        v = self.kp_linear * error_linear

        omega = (
            self.kp_angular * error_angular + self.kp_final_angle * error_final_angle
        )

        self.prev_error_angular = error_angular
        self.prev_error_linear = error_linear

        # ratio = omega / v
        # v = MAX_LINEAR_VELOCITY
        # omega *= ratio

        with self._lock:
            self.linear_velocity = v
            self.angular_velocity = omega

        # Print log
        self.get_logger().info(
            f"Robot Pose: ({robot_pose_x:.2f}, {robot_pose_y:.2f}, {robot_pose_theta:.2f}) "
            f"Object Pose: ({object_pose[0]:.2f}, {object_pose[1]:.2f}, {object_pose[2]:.2f}) "
            f"Desired Pose: ({desired_pose_x:.2f}, {desired_pose_y:.2f}, {desired_pose_theta:.2f}) "
            f"Error: ({error_x:.2f}, {error_y:.2f}, {error_final_angle:.2f}) "
            f"v: {v:.2f} "
            f"omega: {omega:.2f}"
        )

        self.pose_callback_counter = 0

    def controller_callback(self):
        """
        Callback function for the controller
        """
        # Publish the velocity
        if self.pose_callback_counter > 10:
            return

        cmd = Twist()
        with self._lock:
            cmd.linear.x = self.linear_velocity
            cmd.angular.z = self.angular_velocity
            self.publisher.publish(cmd)

        self.pose_callback_counter += 1

    def backwards_callback(self):
        """
        Callback function for the backwards movement
        """
        # Publish the velocity
        if self.pose_callback_counter > 10:
            return

        cmd = Twist()
        with self._lock:
            cmd.linear.x = -MAX_LINEAR_VELOCITY
            cmd.angular.z = 0.0
            self.publisher.publish(cmd)

        self.pose_callback_counter += 1

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

    def get_desired_pose(self, robot_pose: tuple, object_pose: tuple) -> tuple:
        """
        Get desired pose for the robot

        Args:
            robot_pose (tuple): x, y, theta of the robot
            object_pose (tuple): x, y, theta of the object

        Returns:
            tuple: x, y, theta
        """
        # Check at which edge of the object the robot is. The object is a square of 0.42m
        x, y, _ = self.transform_to_frame(robot_pose, object_pose)
        self.get_logger().info(f"{x:.2f}, {y:.2f}")
        radius = 0.4

        # Check on which edge the robot is
        if x > radius:
            x_desired, y_desired, theta_desired = self.transform_to_frame(
                pose=(radius, 0, np.pi),
                transform_frame=(0, 0, 0),
                reference_frame=object_pose,
            )
            self.get_logger().info("x")
        elif x < -radius:
            x_desired, y_desired, theta_desired = self.transform_to_frame(
                pose=(-radius, 0, 0),
                transform_frame=(0, 0, 0),
                reference_frame=object_pose,
            )
            self.get_logger().info("-x")
        elif y > radius:
            x_desired, y_desired, theta_desired = self.transform_to_frame(
                pose=(0, radius, -np.pi / 2),
                transform_frame=(0, 0, 0),
                reference_frame=object_pose,
            )
            self.get_logger().info("y")
        elif y < -radius:
            x_desired, y_desired, theta_desired = self.transform_to_frame(
                pose=(0, -radius, np.pi / 2),
                transform_frame=(0, 0, 0),
                reference_frame=object_pose,
            )
            self.get_logger().info("-y")
        else:
            x_desired, y_desired, theta_desired = robot_pose

        return x_desired, y_desired, theta_desired

    def transform_to_frame(
        self, pose: tuple, transform_frame: tuple, reference_frame: tuple = (0, 0, 0)
    ) -> tuple:
        """
        Transform the pose to the reference frame

        Args:
            pose (tuple): x, y, theta
            transform_frame (tuple): x, y, theta of the frame to transform to
            reference_frame (tuple): x, y, theta of the reference frame

        Returns:
            tuple: x, y, theta
        """

        x, y, theta = pose
        transform_x, transform_y, transform_angle = transform_frame
        reference_x, reference_y, reference_angle = reference_frame

        # Transform to global frame
        x_global = (
            x * np.cos(reference_angle) - y * np.sin(reference_angle)
        ) + reference_x
        y_global = (
            x * np.sin(reference_angle) + y * np.cos(reference_angle)
        ) + reference_y
        theta_global = self.normalize_angle(theta + reference_angle)

        # Transform to the new frame
        x_new = (x_global - transform_x) * np.cos(-transform_angle) - (
            y_global - transform_y
        ) * np.sin(-transform_angle)
        y_new = (x_global - transform_x) * np.sin(-transform_angle) + (
            y_global - transform_y
        ) * np.cos(-transform_angle)
        theta_new = self.normalize_angle(theta_global - transform_angle)

        return x_new, y_new, theta_new

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

    minimal_subscriber = PosePController()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
