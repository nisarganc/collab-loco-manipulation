#!~/ros2-ai/objectpushing/bin python3
# -*- coding: utf-8 -*-

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="pose_estimation",
                executable="aruco_poses_publisher",
                name="aruco_poses_publisher",
                output="screen",
            ),
            Node(
                package="vlm_model",
                executable="scene_perception",
                name="scene_perception",
                output="screen",
            ),
            # Node(
            #     package="optimizer",
            #     executable="optimizer",
            #     name="optimizer",
            #     output="screen",
            # ),
            # Node(
            #     package="vlm_model",
            #     executable="GPT_node",
            #     name="GPT_node",
            #     output="screen",
            # )
        ]
    )
