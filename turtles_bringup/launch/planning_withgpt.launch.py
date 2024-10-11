#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="input",
                executable="llm_text_input_local",
                name="llm_text_input_local",
                output="screen",
            ),
            Node(
                package="llm_model",
                executable="gpt",
                name="gpt",
                output="screen",
            )
        ]
    )
