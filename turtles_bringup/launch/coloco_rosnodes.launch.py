#!~/ros2-ai/objectpushing/bin python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch_ros.actions import Node


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
                package="control",
                executable="p_controller",
                name="p_controller",
                namespace="turtle2",
                output="screen",
            ),
            Node(
                package="control",
                executable="p_controller",
                name="p_controller",
                namespace="turtle4",
                output="screen",
            ),
            Node(
                package="control",
                executable="p_controller",
                name="p_controller",
                namespace="turtle6",
                output="screen",
            )
        ]
    )
