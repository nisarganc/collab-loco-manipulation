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

using namespace std;
std::map<int, std::string> aruco_turtle;

const double ARROW_LENGTH = 0.15;

class ArucoPoseEstimation : public rclcpp::Node {
    public:
        ArucoPoseEstimation() : Node("aruco_poses_publisher") {

            aruco_turtle[0] = "origin";
            aruco_turtle[10] = "turtle2";
            aruco_turtle[20] = "turtle4";
            aruco_turtle[30] = "turtle6";
            aruco_turtle[40] = "object";

            // read camera frames and print fps
            cap = std::make_shared<cv::VideoCapture>(10);
            double fps = cap->get(cv::CAP_PROP_FPS);
            RCLCPP_INFO(this->get_logger(), "Camera FPS: %f", fps);

            if (!cap->isOpened()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open camera");
                return;
            }

            marker_pose_publisher_ = this->create_publisher<msgs_interfaces::msg::MarkerPoseArray>(
                "aruco_poses", 10);
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100), 
                std::bind(&ArucoPoseEstimation::PosesCallback, this));

            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);

            int image_width = 1920;
            int image_height = 1080;

            double fx = 1019.66062; 
            double cx = 944.551199; 
            double fy = 1021.42301; 
            double cy = 460.701976; 
            cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
            distCoeffs = cv::Mat::zeros(1, 5, CV_32F);
            // distCoeffs.at<double>(0) = 0.08621978;
            // distCoeffs.at<double>(1) = 0.08457004;
            // distCoeffs.at<double>(2) = 0.00429467;
            // distCoeffs.at<double>(3) = -0.10166391;
            // distCoeffs.at<double>(4) = -0.06502892;

            bool success = false;
            while (!success) {
                RCLCPP_INFO(this->get_logger(), "Attempting World Frame Origin Detection");
                success = WorldFrame();
            }

            ProjectGoalPose(1.0, 3.0, -1.5);

        }

    private:
        rclcpp::Publisher<msgs_interfaces::msg::MarkerPoseArray>::SharedPtr marker_pose_publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
        std::shared_ptr<cv::VideoCapture> cap;
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        cv::Mat goal_pose, frame, cameraMatrix, distCoeffs, T0, rvec, tvec, Ri, Ti, T_rel, rvec_rel, tvec_rel;
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::Mat projectedGoalPoint2D, projectedArrowEndPoint2D;

        bool WorldFrame() {

            if (!cap->read(frame)) { 
                RCLCPP_WARN(this->get_logger(), "Failed to capture frame");
                return false;
            }

            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);
            
            if (!markerIds.empty()) {

                cv::Mat rvec0, tvec0, R0;

                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.16, cameraMatrix, distCoeffs, rvecs, tvecs);                

                for (size_t i = 0; i < markerIds.size(); ++i) {
                    if (markerIds[i] == 0) {
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

                        return true; 
                    }
                }
            }

            return false;
        }

        void ProjectGoalPose(double x, double y, double yaw) {

            // Convert goal pose to homogeneous coordinates
            cv::Mat goalPoseHomogeneous = cv::Mat::ones(4, 1, CV_64F);
            goalPoseHomogeneous.at<double>(0) = x;
            goalPoseHomogeneous.at<double>(1) = y;
            goalPoseHomogeneous.at<double>(2) = 0; // z

            // Calculate the arrow's end in the world frame
            cv::Mat arrowEndHomogeneous = cv::Mat::ones(4, 1, CV_64F);
            arrowEndHomogeneous.at<double>(0) = x + ARROW_LENGTH * cos(yaw);
            arrowEndHomogeneous.at<double>(1) = y + ARROW_LENGTH * sin(yaw);
            arrowEndHomogeneous.at<double>(2) = 0;

            // Transform the goal pose and arrow end into the camera coordinate system
            cv::Mat goalPoseCamera = T0 * goalPoseHomogeneous;
            cv::Mat arrowEndCamera = T0 * arrowEndHomogeneous;

            // Project these points into the image
            cv::projectPoints(goalPoseCamera.rowRange(0, 3).t(), cv::Mat::zeros(3, 1, CV_64F), 
                            cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distCoeffs, projectedGoalPoint2D);
            cv::projectPoints(arrowEndCamera.rowRange(0, 3).t(), cv::Mat::zeros(3, 1, CV_64F), 
                            cv::Mat::zeros(3, 1, CV_64F), cameraMatrix, distCoeffs, projectedArrowEndPoint2D);
        }


        void PosesCallback() {

            if (!cap->read(frame)) {
                RCLCPP_WARN(this->get_logger(), "Failed to capture frame");
                return;
            }

            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);
            
            if (!markerIds.empty()) {

                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.16, cameraMatrix, distCoeffs, rvecs, tvecs);
                msgs_interfaces::msg::MarkerPoseArray marker_pose_array_msg;

                for (int i = 0; i < markerIds.size(); ++i) {
                    if (markerIds[i] != 0) {

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
                        T_rel = T0.inv() * Ti;  

                        cv::Rodrigues(T_rel.rowRange(0, 3).colRange(0, 3), rvec_rel);
                        tvec_rel = T_rel.rowRange(0, 3).col(3);
                        msgs_interfaces::msg::MarkerPose marker_pose;
                        marker_pose.id = markerIds[i];
                        marker_pose.x = tvec_rel.at<double>(0);
                        marker_pose.y = tvec_rel.at<double>(1);
                        marker_pose.theta = rvec_rel.at<double>(2);
                        marker_pose_array_msg.poses.push_back(marker_pose);

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

                    // Draw the goal pose and yaw arrow
                    cv::Point goalPoint2D(
                        static_cast<int>(projectedGoalPoint2D.at<double>(0)),
                        static_cast<int>(projectedGoalPoint2D.at<double>(1))
                    );
                    cv::Point arrowEndPoint2D(
                        static_cast<int>(projectedArrowEndPoint2D.at<double>(0)),
                        static_cast<int>(projectedArrowEndPoint2D.at<double>(1))
                    );
                    cv::arrowedLine(frame, goalPoint2D, arrowEndPoint2D, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);
                    cv::putText(frame, "Goal", arrowEndPoint2D + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 1,
                                cv::Scalar(0, 0, 255), 2);
                }

                marker_pose_array_msg.image = *cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();

                marker_pose_publisher_->publish(marker_pose_array_msg);
            }

            cv::imshow("Aruco Pose Estimation", frame);
            cv::waitKey(1);
        }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArucoPoseEstimation>());
    rclcpp::shutdown();
    return 0;
}
