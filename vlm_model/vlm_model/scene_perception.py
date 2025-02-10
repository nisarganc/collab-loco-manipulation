# -*- coding: utf-8 -*-

# ROS related
import rclpy
from rclpy.node import Node
from msgs_interfaces.srv import CameraParams
from msgs_interfaces.msg import MarkerCorner, SceneInfo

# LLM related
import json
import os
import time

import cv2
from cv_bridge import CvBridge
import base64


class ScenePerception(Node):
    def __init__(self):
        super().__init__('scene_perception')

        self.cli = self.create_client(CameraParams, 'camera_params')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for cameraparams service')
        response = self.send_request("webcam")

        self.image_width = response.image_width
        self.image_height = response.image_height
        # create cv matrix with fx, fy, cx, cy
        self.camera_matrix = [[response.fx, 0, response.cx], 
                              [0, response.fy, response.cy], 
                              [0, 0, 1]]
        # create T0 4*4 matrix
        self.T0 = [[response.t0[0], response.t0[1], response.t0[2], response.t0[3]],
                   [response.t0[4], response.t0[5], response.t0[6], response.t0[7]],
                   [response.t0[8], response.t0[9], response.t0[10], response.t0[11]],
                   [response.t0[12], response.t0[13], response.t0[14], response.t0[15]]]
                   
        self.scene_info_sub = self.create_subscription(SceneInfo, 'scene_info', self.scene_info_callback, 10)
    

    def send_request(self, camera_name):
        self.req = CameraParams.Request()
        self.req.camera_name = camera_name
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def scene_info_callback(self, msg):
        image = msg.image
        corners = msg.corners

def main(args=None):
    rclpy.init(args=args)
    scene_node = ScenePerception()
    rclpy.spin(scene_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()