import pytest
import numpy as np
import rclpy
from control.p_controller import PosePController

def test_transform_to_frame_1():
    rclpy.init(args=None)
    controller = PosePController()

    # Test case 1: Identity transformation
    pose = (1, 1, np.pi / 4)
    transform_frame = (0, 0, 0)
    reference_frame = (0, 0, 0)
    expected_result = (1, 1, np.pi / 4)
    result = controller.transform_to_frame(pose, transform_frame, reference_frame)
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"

    controller.destroy_node()
    rclpy.shutdown()

def test_transform_to_frame_2():
    rclpy.init(args=None)
    controller = PosePController()

    # Test case 2: Translation only
    pose = (1, 1, 0)
    transform_frame = (1, 1, 0)
    reference_frame = (0, 0, 0)
    expected_result = (0, 0, 0)
    result = controller.transform_to_frame(pose, transform_frame, reference_frame)
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"

    controller.destroy_node()
    rclpy.shutdown()

def test_transform_to_frame_3():
    rclpy.init(args=None)
    controller = PosePController()

    # Test case 3: Rotation only
    pose = (1, 0, 0)
    transform_frame = (0, 0, np.pi / 2)
    reference_frame = (0, 0, 0)
    expected_result = (0, -1, -np.pi / 2)
    result = controller.transform_to_frame(pose, transform_frame, reference_frame)
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"

    controller.destroy_node()
    rclpy.shutdown()

def test_transform_to_frame_4():
    rclpy.init(args=None)
    controller = PosePController()

    # Test case 4: Translation and rotation
    pose = (1, 1, np.pi / 4)
    transform_frame = (1, 1, np.pi / 4)
    reference_frame = (0, 0, 0)
    expected_result = (0, 0, 0)
    result = controller.transform_to_frame(pose, transform_frame, reference_frame)
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"

    controller.destroy_node()
    rclpy.shutdown()

def test_transform_to_frame_5():
    rclpy.init(args=None)
    controller = PosePController()

    # Test case 5: Non-zero reference frame
    pose = (2, 0, 0)
    transform_frame = (0, 0, 0)
    reference_frame = (3, 1, np.pi / 2)
    expected_result = (3, 3, np.pi / 2)
    result = controller.transform_to_frame(pose, transform_frame, reference_frame)
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"

    controller.destroy_node()
    rclpy.shutdown()

def test_transform_to_frame_6():
    rclpy.init(args=None)
    controller = PosePController()

    # Test case 5: Non-zero reference frame
    pose = (2, 0, np.pi / 2)
    transform_frame = (0, 0, 0)
    reference_frame = (3, 1, np.pi / 2)
    expected_result = (3, 3, -np.pi)
    result = controller.transform_to_frame(pose, transform_frame, reference_frame)
    assert np.allclose(result, expected_result), f"Expected {expected_result}, got {result}"

    controller.destroy_node()
    rclpy.shutdown()

    

if __name__ == "__main__":
    pytest.main()