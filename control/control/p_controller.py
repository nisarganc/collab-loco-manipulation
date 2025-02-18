import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading

from geometry_msgs.msg import Twist
from msgs_interfaces.msg import MarkerPoseArray

MAX_LINEAR_VELOCITY = 0.05  # m/s
MAX_ANGULAR_VELOCITY = 0.05  # rad/s
RADIUS = 0.45  # m

IDs = {"/turtle2": 10, "/turtle4": 20, "/turtle6": 30, "object": 40}

desired = {
    "/turtle2": (3.072860444107310407e-01, -9.158966649116762060e-02),
    "/turtle4": (-3.262809077883643827e-02, -3.881981339339535264e-01),
    "/turtle6": (-2.775603127609371779e-01, 3.278249424045180938e-01),
}


class PosePController(Node):
    """
    Class for the Pose P Controller

    Attributes:
        kp_angular (float): Proportional gain for angular velocity
        kp_linear (float): Proportional gain for linear velocity
        kp_final_angle (float): Proportional gain for the final angle
        dt (float): Time step
        linear_velocity (float): Linear velocity
        angular_velocity (float): Angular velocity
        segment (list): Segment for the desired coordinates
        subscription (Subscriber): Subscriber for the Aruco marker
        publisher (Publisher): Publisher for the cmd_vel
        timer (Timer): Timer for the controller
        pose_callback_counter (int): Counter for the pose callback
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
        self.kp_linear = 1.0
        self.kp_angular = 6.0
        self.kp_final_angle = 0.0

        # Time step
        self.dt = 0.1  # 100ms

        # Velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Segment
        self.segment = []
        self.desired_pose = self.get_desired_pose()

        # callback group
        callback_group = ReentrantCallbackGroup()

        # Create Client for segment arrays
        # self.segment_client = self.create_client(Segment, "")

        # Create Subscriber for Aruco
        self.pose_subscriber = self.create_subscription(
            MarkerPoseArray,
            "/aruco_poses",
            self.handle_pose_update_callback,
            10,
            callback_group=callback_group,
        )

        # Create Publisher for cmd_vel and Timer
        self.cmd_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.timer = self.create_timer(
            self.dt, self.publish_velocity_callback, callback_group=callback_group
        )
        self.pose_callback_counter = 0

    def handle_pose_update_callback(self, msg: MarkerPoseArray):
        """
        Callback function for the subscriber to handle the pose update

        Args:
            msg (Pose): Pose message from the Aruco marker
        """
        # Get the poses
        object_pose = self.get_pose(msg, IDs["object"])
        robot_pose = self.get_pose(msg, IDs[self.get_namespace()])

        if None in robot_pose or None in object_pose:
            return

        # if self.segment == []:
        #     self.segment = self.generate_desired_segment(robot_pose)
        #     self.get_logger().info(f"Desired Segment Coordinates: {self.segment}")
        #     if self.segment == []:
        #         return
        # self.desired_pose = self.get_desired_pose()

        v, omega = self.p_controller(robot_pose)
        v, omega = self.check_limits(v, omega)

        with self._lock:
            self.linear_velocity = v
            self.angular_velocity = omega

        self.pose_callback_counter = 0

    def publish_velocity_callback(self):
        """
        Callback function for the timer to publish the velocity
        """
        if self.pose_callback_counter > 10:
            return

        cmd = Twist()
        with self._lock:
            cmd.linear.x = self.linear_velocity
            cmd.angular.z = self.angular_velocity

        self.cmd_publisher.publish(cmd)
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

    def get_desired_pose(self) -> tuple:
        """
        Get desired pose for the robot in the object frame

        Returns:
            tuple: x, y, theta
        """
        # x_desired, y_desired = self.segment[len(self.segment) // 2]  # Middle point of the segment
        x_desired, y_desired = desired[self.get_namespace()]
        theta_desired = self.normalize_angle(np.arctan2(y_desired, x_desired) + np.pi)

        return x_desired, y_desired, theta_desired

    def p_controller(self, robot_pose: tuple) -> tuple:
        """
        P Controller for the robot

        Args:
            robot_pose (tuple): x, y, theta of the robot in the object frame

        Returns:
            tuple: Linear velocity, Angular velocity
        """
        robot_pose_x, robot_pose_y, robot_pose_theta = robot_pose
        desired_pose_x, desired_pose_y, desired_pose_theta = self.desired_pose

        # Calculate the errors
        error_x = desired_pose_x - robot_pose_x
        error_y = desired_pose_y - robot_pose_y

        error_final_angle = self.shortest_angular_distance(
            robot_pose_theta, desired_pose_theta
        )
        error_linear = np.sqrt(error_x**2 + error_y**2)
        error_angular = self.normalize_angle(
            np.arctan2(error_y, error_x) - robot_pose_theta
        )

        # Calculate the velocities
        v = self.kp_linear * error_linear
        omega = (
            self.kp_angular * error_angular + self.kp_final_angle * error_final_angle
        )

        # Print log
        self.get_logger().info(
            f"Robot Pose: ({robot_pose_x:.2f}, {robot_pose_y:.2f}, {robot_pose_theta:.2f}) "
            f"Desired Pose: ({desired_pose_x:.2f}, {desired_pose_y:.2f}, {desired_pose_theta:.2f}) "
            f"Error: ({error_x:.2f}, {error_y:.2f}, {error_final_angle:.2f}) "
            f"v: {v:.2f} "
            f"omega: {omega:.2f}"
        )
        self.save_plot_data(robot_pose)

        return v, omega

    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle

        Args:
            angle (float): Angle to normalize

        Returns:
            float: Normalized angle
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def shortest_angular_distance(self, x, y):
        """
        Calculate the shortest angular distance between two angles

        Args:
            x (float): First angle
            y (float): Second angle

        Returns:
            float: Shortest angular
        """
        return min(y - x, y - x + 2 * np.pi, y - x - 2 * np.pi, key=abs)

    def check_limits(self, v: float, omega: float) -> tuple:
        """
        Check the limits of the velocities

        Args:
            v (float): Linear velocity
            omega (float): Angular velocity

        Returns:
            tuple: Linear velocity, Angular velocity
        """
        if v > MAX_LINEAR_VELOCITY:
            ratio = abs(MAX_LINEAR_VELOCITY / v)
            v = MAX_LINEAR_VELOCITY
            omega = omega * ratio

        elif v < -MAX_LINEAR_VELOCITY:
            ratio = abs(MAX_LINEAR_VELOCITY / v)
            v = -MAX_LINEAR_VELOCITY
            omega = omega * ratio

        if omega > MAX_ANGULAR_VELOCITY:
            ratio = abs(MAX_ANGULAR_VELOCITY / omega)
            omega = MAX_ANGULAR_VELOCITY
            v = v * ratio

        elif omega < -MAX_ANGULAR_VELOCITY:
            ratio = abs(MAX_ANGULAR_VELOCITY / omega)
            omega = -MAX_ANGULAR_VELOCITY
            v = v * ratio

        return v, omega

    def generate_desired_segment(
        self,
        robot_pose: tuple,
        radius: float = RADIUS,
        num_of_points: int = 3,
        num_of_segments: int = 4,
    ) -> list:
        """
        Generate the desired coordinates for the robot as a segment

        Args:
            robot_pose (tuple): x, y, theta of the robot in the object frame
            radius (float): Radius of the object
            num_of_points (int): Number of points to generate
            num_of_segments (int): Number of segments

        Returns:
            list: List of coordinates in the object frame in the segment
        """
        robot_x, robot_y, _ = robot_pose
        corners = [
            (radius, radius),
            (radius, -radius),
            (-radius, -radius),
            (-radius, radius),
        ]

        coordinates = []
        segment = []

        for edge in range(4):
            p1 = corners[edge]
            p2 = corners[(edge + 1) % 4]

            for i in range(num_of_points):
                x = p1[0] + (p2[0] - p1[0]) * i / (num_of_points - 1)
                y = p1[1] + (p2[1] - p1[1]) * i / (num_of_points - 1)
                coordinates.append([x, y])

        segment_id = -1

        # Check on which edge the robot is
        if robot_x > radius:
            segment_id = 0
        elif robot_x < -radius:
            segment_id = 2
        elif robot_y > radius:
            segment_id = 3
        elif robot_y < -radius:
            segment_id = 1

        num_per_segment = (num_of_points * 4) // num_of_segments

        if segment_id == -1:
            return segment

        for j in range(num_per_segment):
            segment.append(coordinates[segment_id * num_per_segment + j])

        return segment

    def save_plot_data(self, robot_pose: tuple):
        """
        Save the plot data to a file
        """
        namespace = self.get_namespace().split("/")[1]
        x, y, theta = robot_pose

        # Add robot pose to the csv file
        with open(f"robot_poses_{namespace}.csv", "a") as f:
            np.savetxt(f, [[x, y, theta]], delimiter=",", fmt="%.2f")


def main(args=None):
    rclpy.init(args=args)

    controller = PosePController()

    executor = MultiThreadedExecutor()
    executor.add_node(controller)

    # while not controller.segment_client.wait_for_service(timeout_sec=1.0):
    #     controller.get_logger().info('Service not available, waiting again...')

    # request = Segment.Request()
    # future = controller.segment_client.call_async(request)
    # rclpy.spin_until_future_complete(controller, future)

    # if future.result() is not None:
    #     controller.get_logger().info('Service call succeeded')
    # else:
    #     controller.get_logger().error('Service call failed')

    try:
        executor.spin()
    except KeyboardInterrupt:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
