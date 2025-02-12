import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import SceneInfo
import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')
        self.subscription = self.create_subscription(
            SceneInfo,
            '/aruco_poses',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # show image
        cv2.imshow('image', msg)


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()