
// 1. Reads aruco_poses msg and stores them in turtle2, turtle4, turtle6, centroid_pose, and frame.
// 2. Draws the trajectory on the frame if traj_generated is true.

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include "std_msgs/msg/string.hpp"
#include <cv_bridge/cv_bridge.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <string>

#include <msgs_interfaces/msg/marker_pose.hpp>
#include <msgs_interfaces/msg/marker_pose_array.hpp>

#include <msgs_interfaces/msg/scene_info.hpp>
#include <msgs_interfaces/srv/camera_params.hpp>

using std::placeholders::_1;
const double ARROW_LENGTH = 0.05;

struct ROSPose {
  double x;
  double y;
  double yaw;
};

ROSPose turtle2, turtle4, turtle6, centroid_pose;

class ArucoPoseSubscriber : public rclcpp::Node {
    public:
        ArucoPoseSubscriber() : Node("aruco_pose_subscriber") {
            // create subscriber to the marker poses
            marker_pose_subscriber_ = this->create_subscription<msgs_interfaces::msg::MarkerPoseArray>(
                "aruco_poses", 10, std::bind(&ArucoPoseSubscriber::MarkerPosesCallback, this, _1));    
        }

    private:
        rclcpp::Subscription<msgs_interfaces::msg::MarkerPoseArray>::SharedPtr marker_pose_subscriber_;

        void MarkerPosesCallback(const msgs_interfaces::msg::MarkerPoseArray::SharedPtr msg) {

            for (int i = 0; i < msg->poses.size(); i++) {
                if (msg->poses[i].id == 10) {
                    turtle2.x = msg->poses[i].x;
                    turtle2.y = msg->poses[i].y;
                    turtle2.yaw = msg->poses[i].theta;
                }
                else if (msg->poses[i].id == 20) {
                    turtle4.x = msg->poses[i].x;
                    turtle4.y = msg->poses[i].y;
                    turtle4.yaw = msg->poses[i].theta;
                }
                else if (msg->poses[i].id == 30) {
                    turtle6.x = msg->poses[i].x;
                    turtle6.y = msg->poses[i].y;
                    turtle6.yaw = msg->poses[i].theta;
                }
                else if (msg->poses[i].id == 40) {
                    centroid_pose.x = msg->poses[i].x;
                    centroid_pose.y = msg->poses[i].y;
                    centroid_pose.yaw = msg->poses[i].theta;
                }
            }
        }
};


cv::Mat frame;
bool mark_goal = true;
double goal_x = 1.0;
double goal_y = 3.0;
double goal_yaw = -1.5;
bool traj_generated = false;
std::vector<std::tuple<double, double, double>> traj_pose_2d;
class SceneInfoSubscriber : public rclcpp::Node {
    public:
        SceneInfoSubscriber() : Node("scene_info_subscriber") {

        // create client to the camera params service
        camera_params_client_ = this->create_client<msgs_interfaces::srv::CameraParams>("camera_params");
        auto request = std::make_shared<msgs_interfaces::srv::CameraParams::Request>();
        request->camera_name = "webcam";

        while (!camera_params_client_->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_INFO(this->get_logger(), "waiting for camera params service");
        }

        auto result = camera_params_client_->async_send_request(request);
        // wait for the result
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result) == 
            rclcpp::FutureReturnCode::SUCCESS) {
            auto response = result.get();
            cameraMatrix = (cv::Mat_<double>(3,3) << response->fx, 0.0, response->cx, 0.0, response->fy, response->cy, 0.0, 0.0, 1.0);
            distCoeffs = cv::Mat::zeros(1, 5, CV_32F);
            T0 = (cv::Mat_<double>(4, 4) << response->t0[0], response->t0[1], response->t0[2], response->t0[3], 
                                            response->t0[4], response->t0[5], response->t0[6], response->t0[7], 
                                            response->t0[8], response->t0[9], response->t0[10], response->t0[11], 
                                            response->t0[12], response->t0[13], response->t0[14], response->t0[15]);
        }
        else {
            RCLCPP_ERROR(this->get_logger(), "Failed to get camera params");
        }

        scene_info_subscriber_ = this->create_subscription<msgs_interfaces::msg::SceneInfo>(
            "scene_info", 10, std::bind(&SceneInfoSubscriber::SceneInfoCallback, this, _1));
    
        }

    private:
        rclcpp::Client<msgs_interfaces::srv::CameraParams>::SharedPtr camera_params_client_;
        rclcpp::Subscription<msgs_interfaces::msg::SceneInfo>::SharedPtr scene_info_subscriber_;
        cv::Mat T0, cameraMatrix, distCoeffs;

        void SceneInfoCallback(const msgs_interfaces::msg::SceneInfo::SharedPtr msg) {
            // read frame from sensor msg
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
            frame = cv_ptr->image;

            if (mark_goal) {
            // Convert goal pose to homogeneous coordinates
            cv::Mat goalPoseHomogeneous = cv::Mat::ones(4, 1, CV_64F);
            goalPoseHomogeneous.at<double>(0) = goal_x;
            goalPoseHomogeneous.at<double>(1) = goal_y;
            goalPoseHomogeneous.at<double>(2) = 0; // z

            // Calculate the arrow's end in the world frame
            cv::Mat arrowEndHomogeneous = cv::Mat::ones(4, 1, CV_64F);
            arrowEndHomogeneous.at<double>(0) = goal_x + ARROW_LENGTH * cos(goal_yaw);
            arrowEndHomogeneous.at<double>(1) = goal_y + ARROW_LENGTH * sin(goal_yaw);
            arrowEndHomogeneous.at<double>(2) = 0;

            // Transform the goal pose and arrow end into the camera coordinate system
            cv::Mat goalPoseCamera = T0 * goalPoseHomogeneous;
            cv::Mat arrowEndCamera = T0 * arrowEndHomogeneous;
            cv::Mat projectedGoalPoint2D, projectedArrowEndPoint2D;

            // Project these points into the image
            cv::projectPoints(goalPoseCamera.rowRange(0, 3).t(), cv::Mat::zeros(3, 1, CV_64F), 
                            cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distCoeffs, projectedGoalPoint2D);
            cv::projectPoints(arrowEndCamera.rowRange(0, 3).t(), cv::Mat::zeros(3, 1, CV_64F), 
                            cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distCoeffs, projectedArrowEndPoint2D);
            
            
            // Draw the goal pose and yaw arrow
            cv::Point goalPoint2D(
                static_cast<int>(projectedGoalPoint2D.at<double>(0)),
                static_cast<int>(projectedGoalPoint2D.at<double>(1))
            );
            cv::Point arrowEndPoint2D(
                static_cast<int>(projectedArrowEndPoint2D.at<double>(0)),
                static_cast<int>(projectedArrowEndPoint2D.at<double>(1))
            );
            cv::circle(frame, goalPoint2D, 2, cv::Scalar(255, 0, 0), 6);
            cv::arrowedLine(frame, goalPoint2D, arrowEndPoint2D, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);
            cv::putText(frame, "Goal", arrowEndPoint2D + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(0, 0, 255), 2);
            }
            

            if (traj_generated) {
                std::cout << "Drawing trajectory" << std::endl;
                for (const auto& [x, y, yaw] : traj_pose_2d) {
                    cv::Mat trajPointHomogeneous = cv::Mat::ones(4, 1, CV_64F);
                    trajPointHomogeneous.at<double>(0) = x;
                    trajPointHomogeneous.at<double>(1) = y;
                    trajPointHomogeneous.at<double>(2) = 0;  // z

                    cv::Mat trajEndPointHomogeneous = cv::Mat::ones(4, 1, CV_64F);
                    trajEndPointHomogeneous.at<double>(0) = x + ARROW_LENGTH * cos(yaw);
                    trajEndPointHomogeneous.at<double>(1) = y + ARROW_LENGTH * sin(yaw);
                    trajEndPointHomogeneous.at<double>(2) = 0;

                    cv::Mat trajPointCamera = T0 * trajPointHomogeneous;
                    cv::Mat trajEndPointCamera = T0 * trajEndPointHomogeneous;
                    cv::Mat projectedTrajPoint2D, projectedTrajEndPoint2D;

                    cv::projectPoints(trajPointCamera.rowRange(0, 3).t(), cv::Mat::zeros(3, 1, CV_64F),
                                    cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distCoeffs, projectedTrajPoint2D);
                    cv::projectPoints(trajEndPointCamera.rowRange(0, 3).t(), cv::Mat::zeros(3, 1, CV_64F),
                                    cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distCoeffs, projectedTrajEndPoint2D); 

                    cv::Point trajPoint2D(
                        static_cast<int>(projectedTrajPoint2D.at<double>(0)),
                        static_cast<int>(projectedTrajPoint2D.at<double>(1))
                    );
                    cv::Point trajEndPoint2D(
                        static_cast<int>(projectedTrajEndPoint2D.at<double>(0)),
                        static_cast<int>(projectedTrajEndPoint2D.at<double>(1))
                    );

                    cv::circle(frame, trajPoint2D, 2, cv::Scalar(255, 0, 0), 6);
                    cv::arrowedLine(frame, trajPoint2D, trajEndPoint2D, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);   
                }
            }

            cv::imshow("Aruco Markers", frame);
            cv::waitKey(0);
        }
};
