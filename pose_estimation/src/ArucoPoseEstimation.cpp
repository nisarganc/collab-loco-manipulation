// 1. Projects robots' poses, object pose, and goal pose onto the camera frame.
// 2. Publishes poses array and frame every 100ms. 

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tf2/LinearMath/Quaternion.h>
#include <vector>
#include <iostream>
#include <cmath>

#include <map>
#include <string>

#include <msgs_interfaces/msg/marker_pose.hpp>
#include <msgs_interfaces/msg/marker_pose_array.hpp>

#include <msgs_interfaces/msg/marker_point.hpp>
#include <msgs_interfaces/msg/scene_info.hpp>

#include <msgs_interfaces/srv/camera_params.hpp>

using namespace std;
std::map<int, std::string> aruco_turtle;

const double ARROW_LENGTH = 0.05;
const int image_height = 1080; //rows
const int image_width = 1920; //columns
const double fx = 1019.66062; 
const double cx = 944.551199; 
const double fy = 1021.42301; 
const double cy = 460.701976;
// distCoeffs.at<double>(0) = 0.08621978;
// distCoeffs.at<double>(1) = 0.08457004;
// distCoeffs.at<double>(2) = 0.00429467;
// distCoeffs.at<double>(3) = -0.10166391;
// distCoeffs.at<double>(4) = -0.06502892;

class ArucoPoseEstimation : public rclcpp::Node {
    public:
        ArucoPoseEstimation() : Node("aruco_poses_publisher") {

            aruco_turtle[0] = "origin";
            aruco_turtle[10] = "turtle2";
            aruco_turtle[20] = "turtle4";
            aruco_turtle[30] = "turtle6";
            aruco_turtle[40] = "object";

            cap = std::make_shared<cv::VideoCapture>(6);
            // cap.set(cv::CAP_PROP_FRAME_HEIGHT, image_height);            
            // cap.set(cv::CAP_PROP_FRAME_WIDTH, image_width);
            // cap.set(cv::CAP_PROP_FPS, 5);

            if (!cap->isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open camera");
                return;
            }

            marker_pose_publisher_ = this->create_publisher<msgs_interfaces::msg::MarkerPoseArray>(
                "aruco_poses", 10);
            scene_publisher_ = this->create_publisher<msgs_interfaces::msg::SceneInfo>(
                "scene_info", 10);
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100), 
                std::bind(&ArucoPoseEstimation::PosesCallback, this));

            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
 
            cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
            distCoeffs = cv::Mat::zeros(1, 5, CV_32F);

            bool success = false;
            while (!success) {
                RCLCPP_INFO(this->get_logger(), "Attempting World Frame Origin Detection");
                success = WorldFrame();
            }

            camera_params_service_ = this->create_service<msgs_interfaces::srv::CameraParams>(
                "camera_params_service", std::bind(&ArucoPoseEstimation::CameraParamsCallback, 
                this, std::placeholders::_1, std::placeholders::_2));

        }

    private:
        rclcpp::Publisher<msgs_interfaces::msg::MarkerPoseArray>::SharedPtr marker_pose_publisher_;

        msgs_interfaces::msg::MarkerPoint marker0_point;
        rclcpp::Publisher<msgs_interfaces::msg::SceneInfo>::SharedPtr scene_publisher_;

        rclcpp::Service<msgs_interfaces::srv::CameraParams>::SharedPtr camera_params_service_;

        rclcpp::TimerBase::SharedPtr timer_;
        std::shared_ptr<cv::VideoCapture> cap;
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        cv::Mat goal_pose, frame, cameraMatrix, distCoeffs, T0, T4, rvec, tvec, Ri, Ti, T_rel, rvec_rel, tvec_rel;
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<cv::Vec3d> rvecs, tvecs;
        std::vector<double> T0_values;

        bool WorldFrame() {

            if (!cap->read(frame)) { 
                RCLCPP_WARN(this->get_logger(), "Failed to capture frame");
                return false;
            }
            // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);

            if (!markerIds.empty()) {

                cv::Mat rvec0, tvec0, R0;

                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.16, cameraMatrix, distCoeffs, rvecs, tvecs);                
                for (size_t i = 0; i < markerIds.size(); ++i) {
                    if (markerIds[i] == 0) {

                        marker0_point.id = 0;
                        cv::Point2f DetectedPoint = (markerCorners[i][0] + markerCorners[i][1] 
                                                    + markerCorners[i][2] + markerCorners[i][3]) / 4;

                        marker0_point.centre_point.x = DetectedPoint.x;
                        marker0_point.centre_point.y = DetectedPoint.y;
                        
                        rvec0 = cv::Mat(rvecs[i]);
                        tvec0 = cv::Mat(tvecs[i]);

                        // compute rotation matrix
                        cv::Rodrigues(rvec0, R0);

                        // create 4*4 transformation matrix
                        T0 = cv::Mat::eye(4, 4, CV_64F);

                        // Fill in rotation and translation for m0Xc
                        R0.copyTo(T0.rowRange(0, 3).colRange(0, 3));
                        tvec0.copyTo(T0.rowRange(0, 3).col(3));
                        T0.at<double>(3, 0) = 0;
                        T0.at<double>(3, 1) = 0;
                        T0.at<double>(3, 2) = 0;

                        // initialize matrices
                        rvec = cv::Mat::zeros(3, 1, CV_64F);
                        tvec = cv::Mat::zeros(3, 1, CV_64F);            
                        Ri = cv::Mat::zeros(3, 3, CV_64F);
                        Ti = cv::Mat::eye(4, 4, CV_64F);
                        T_rel = cv::Mat::eye(4, 4, CV_64F);
                        rvec_rel = cv::Mat::zeros(3, 1, CV_64F);
                        tvec_rel = cv::Mat::zeros(3, 1, CV_64F);

                        RCLCPP_INFO(this->get_logger(), "World Frame Found");

                        for (int i = 0; i < T0.rows; i++) {
                            for (int j = 0; j < T0.cols; j++) {
                                T0_values.push_back(T0.at<double>(i, j));
                            }
                        }

                        return true; 
                    }
                }
            }

            return false;
        }

        void CameraParamsCallback(
            const std::shared_ptr<msgs_interfaces::srv::CameraParams::Request> request,
            std::shared_ptr<msgs_interfaces::srv::CameraParams::Response> response) {

            response->image_width = image_width;
            response->image_height = image_height;

            response->fx = fx;
            response->fy = fy;
            response->cx = cx;
            response->cy = cy;

            for (int i = 0; i < T0_values.size(); i++) {
                response->t0[i] = T0_values[i];
            }

        }


        void PosesCallback() {

            if (!cap->read(frame)) {
                RCLCPP_WARN(this->get_logger(), "Failed to capture frame");
                return;
            }
            // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);            
            
            if (!markerIds.empty()) {

                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.16, cameraMatrix, distCoeffs, rvecs, tvecs);
                msgs_interfaces::msg::MarkerPoseArray marker_pose_array_msg;
                msgs_interfaces::msg::SceneInfo scene_msg;

                scene_msg.marker_points.push_back(marker0_point);
                for (int i = 0; i < markerIds.size(); ++i)
                {
                    if (markerIds[i] == 40) {
                        rvec = cv::Mat(rvecs[i]);
                        tvec = cv::Mat(tvecs[i]);

                        // Compute rotation matrix
                        cv::Rodrigues(rvec, Ri);
                        T4 = cv::Mat::eye(4, 4, CV_64F);

                        // Fill in rotation and translation for m0Xc and m1Xc
                        Ri.copyTo(T4.rowRange(0, 3).colRange(0, 3));
                        tvec.copyTo(T4.rowRange(0, 3).col(3));
                        T4.at<double>(3, 0) = 0;
                        T4.at<double>(3, 1) = 0;
                        T4.at<double>(3, 2) = 0;

                        // Compute relative transformation
                        T_rel = T0.inv() * Ti;

                        cv::Rodrigues(T_rel.rowRange(0, 3).colRange(0, 3), rvec_rel);
                        tvec_rel = T_rel.rowRange(0, 3).col(3);
                        
                        msgs_interfaces::msg::MarkerPose marker_pose;
                        marker_pose.id = markerIds[i];
                        marker_pose.x = tvec_rel.at<double>(0);
                        marker_pose.y = tvec_rel.at<double>(1);
                        marker_pose.theta = rvec_rel.at<double>(2);
                        marker_pose_array_msg.poses.push_back(marker_pose);

                        msgs_interfaces::msg::MarkerPoint marker_point;
                        marker_point.id = markerIds[i];
                        cv::Point2f detected_point = ( markerCorners[i][0] + markerCorners[i][1] 
                                        + markerCorners[i][2] + markerCorners[i][3] ) / 4;   
                        marker_point.centre_point.x = detected_point.x;
                        marker_point.centre_point.y = detected_point.y;                       
                        
                        scene_msg.marker_points.push_back(marker_point);
                    }
                }

                for (int i = 0; i < markerIds.size(); ++i) {

                    if (markerIds[i] != 0 && markerIds[i] != 40) {

                        rvec = cv::Mat(rvecs[i]);
                        tvec = cv::Mat(tvecs[i]);

                        // Compute rotation matrix
                        cv::Rodrigues(rvec, Ri);

                        // Fill in rotation and translation for m0Xc and m1Xc
                        Ri.copyTo(Ti.rowRange(0, 3).colRange(0, 3));
                        tvec.copyTo(Ti.rowRange(0, 3).col(3));
                        Ti.at<double>(3, 0) = 0;
                        Ti.at<double>(3, 1) = 0;
                        Ti.at<double>(3, 2) = 0;

                        // Compute relative transformation
                        T_rel = T4.inv() * Ti;  

                        cv::Rodrigues(T_rel.rowRange(0, 3).colRange(0, 3), rvec_rel);
                        tvec_rel = T_rel.rowRange(0, 3).col(3);
                        msgs_interfaces::msg::MarkerPose marker_pose;
                        marker_pose.id = markerIds[i];
                        marker_pose.x = tvec_rel.at<double>(0);
                        marker_pose.y = tvec_rel.at<double>(1);
                        marker_pose.theta = rvec_rel.at<double>(2);
                        marker_pose_array_msg.poses.push_back(marker_pose);

                        msgs_interfaces::msg::MarkerPoint marker_point;
                        marker_point.id = markerIds[i];
                        cv::Point2f detected_point = ( markerCorners[i][0] + markerCorners[i][1] 
                                        + markerCorners[i][2] + markerCorners[i][3] ) / 4;   
                        marker_point.centre_point.x = detected_point.x;
                        marker_point.centre_point.y = detected_point.y;                       
                        
                        scene_msg.marker_points.push_back(marker_point);

                        // RCLCPP_INFO(this->get_logger(), "marker: %d %f %f %f", markerIds[i], tvec_rel.at<double>(0), tvec_rel.at<double>(1), rvec_rel.at<double>(2));
                    }
                
                    // cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
                    // cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);

                    // custom detection marking
                    cv::Point2f redPoint = markerCorners[i][3]; 
                    cv::Point2f cornerPointx = markerCorners[i][2]; 
                    cv::Point2f cornerPointy = markerCorners[i][0]; 
                    cv::Point2f otherPoint = markerCorners[i][1];

                    // calculate the centre point the border redpoint, cornerPointx, cornerPointy, otherPoint
                    cv::Point2f centrePoint = (redPoint + cornerPointx + cornerPointy + otherPoint) / 4;
                    cv::Point2f midy = (cornerPointy + otherPoint) / 2;
                    cv::Point2f midx = (cornerPointx + otherPoint) / 2;

                    // draw a red x, green y line, and a blue point for aruco markers 
                    cv::arrowedLine(frame, centrePoint, midx, cv::Scalar(0, 0, 255), 3);
                    // cv::arrowedLine(frame, centrePoint, midy, cv::Scalar(0, 255, 0), 3);
                    cv::circle(frame, centrePoint, 2, cv::Scalar(255, 0, 0), 6);
                    cv::putText(frame, aruco_turtle[markerIds[i]], cornerPointy, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                }

                marker_pose_publisher_->publish(marker_pose_array_msg);

                // cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
                scene_msg.image = *cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
                scene_publisher_->publish(scene_msg);
                
            }   
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArucoPoseEstimation>());
    rclcpp::shutdown();
    return 0;
}
