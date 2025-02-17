# -*- coding: utf-8 -*-

# ROS related
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Custom messages
from msgs_interfaces.srv import CameraParams
from msgs_interfaces.msg import SceneInfo

# AI model related
import torch
from PIL import Image as PILImage
from transformers import pipeline
from pathlib import Path
from sam2.build_sam import build_sam2_camera_predictor

# helper libraries
import cv2
from cv_bridge import CvBridge
import numpy as np


class ScenePerception(Node):
    def __init__(self):
        super().__init__('scene_perception')

        # Get camera parameters
        self.cli = self.create_client(CameraParams, 'camera_params_service')

        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('waiting for camera_params_service')
        response = self.send_request("webcam")

        self.image_height = response.image_height
        self.image_width = response.image_width
        self.camera_matrix = [[response.fx, 0, response.cx], 
                              [0, response.fy, response.cy], 
                              [0, 0, 1]]
        self.t0 = np.array(response.t0).reshape(4, 4)
        self.t0 = np.linalg.inv(self.t0) # camera to world coordinate system

        # initialize ai models
        self.first = True
        self.grounding_dino = pipeline(model="IDEA-Research/grounding-dino-tiny", 
                                task="zero-shot-object-detection", 
                                device="cuda")
 
        self.sam2 = build_sam2_camera_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",
                                                "/home/nisarganc/segment_anything/checkpoints/sam2.1_hiera_small.pt"
                                                )

        # sub and pub scene information
        self.cv_bridge = CvBridge()
        self.scene_info_sub = self.create_subscription(SceneInfo, 'scene_info', self.detect_segment_track, 10)
        self.annotated_frame_pub = self.create_publisher(Image, 'annotated_frame', 10)
    

    def send_request(self, camera_name):
        self.req = CameraParams.Request()
        self.req.camera_name = camera_name
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def detect_segment_track(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg.image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        gd_image = PILImage.fromarray(cv_image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            
            if self.first:
                # self.first = False
                
                dino_output = self.grounding_dino(gd_image,  candidate_labels=["objects."], threshold=0.3)
                # [{'score': 0.74167400598526, 'label': 'objects.', 'box': {'xmin': 644, 'ymin': 570, 'xmax': 1122, 'ymax': 1033}}, 
                # {'score': 0.5053098797798157, 'label': 'objects.', 'box': {'xmin': 1772, 'ymin': 909, 'xmax': 1884, 'ymax': 1047}}, 
                # {'score': 0.3090237081050873, 'label': 'objects.', 'box': {'xmin': 808, 'ymin': 734, 'xmax': 959, 'ymax': 891}}]

                obstacles_bboxes = []
                object_bbox = []
                for output in dino_output:
                    bbox = [output['box']['xmin'], output['box']['ymin'], output['box']['xmax'], output['box']['ymax']]
                    found = False
                    for marker_point in msg.marker_points:
                        if (marker_point.centre_point.x > bbox[0] and marker_point.centre_point.x < bbox[2] and 
                            marker_point.centre_point.y > bbox[1] and marker_point.centre_point.y < bbox[3]):
                            if marker_point.id == 40:
                                object_bbox = bbox
                            found = True

                    if not found:
                        obstacles_bboxes.append(bbox)  

                obstacles_bboxes = np.array(obstacles_bboxes)  
                object_bbox = np.array(object_bbox)      
                            
                self.sam2.load_first_frame(cv_image)
                # for i, bbox in enumerate(obstacles_bboxes):
                    # _, out_obj_ids, video_res_masks = self.sam2.add_new_prompt(
                    #                                                     frame_idx=0, 
                    #                                                     obj_id=i,
                    #                                                     bbox=bbox)
                _, _, object_mask = self.sam2.add_new_prompt(
                                                            frame_idx=0, 
                                                            obj_id=40,
                                                            bbox=object_bbox)  

                # ToDO: obtain polygon countour of the object
                object_mask = object_mask[0].cpu().float() # (num_objects, C, H, W)-> (C, H, W)
                object_mask = object_mask.permute(1, 2, 0)
                object_mask = object_mask.mean(axis=-1)
                object_mask = (object_mask > 0).int()
                object_mask = object_mask.numpy().astype(np.uint8)

                mask_uint8 = (object_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # IMAGE FRAME
                contact_points = np.array(contours)
                contact_points = np.squeeze(contact_points) # (num_points, 2)

                t4 = np.array(msg.object_frame).reshape(4, 4)
                t4 = np.linalg.inv(t4) # camera to object coordinate system

                # Micheal's Math!!
                zm = np.array([t4[0,2], t4[1,2], t4[2,2]]) # 1x3
                o = np.array([t4[0,3], t4[1,3], t4[2,3]]) # 1x3
                d = - np.dot(zm, o.T) # 1x1

                # convert contact points to camera coordinate system
                contact_points = np.concatenate((contact_points, np.ones((contact_points.shape[0], 1))), axis=1)
                contact_points = np.dot(np.linalg.inv(self.camera_matrix), contact_points.T)

                # calculate scaling factor
                s = -d/np.dot(zm, contact_points) # num_points,

                s = np.expand_dims(s, axis=1) # (num_points, 1)
                contact_points = contact_points.T # (num_points, 3)
                contact_points = s * contact_points # (num_points, 3)

                # convert contact points to object coordinate system
                contact_points = np.concatenate((contact_points, np.ones((contact_points.shape[0], 1))), axis=1)
                contact_points = np.dot(t4, contact_points.T).T
                contact_points = contact_points[:, :2] # (num_points, 2)

                for bbox in obstacles_bboxes:
                    cv_image = cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) 
                cv2.drawContours(cv_image, contours, -1, (255, 255, 0), 2)

                # Convert obstacles_bboxes to T0

            # else:
            #     out_obj_ids, video_res_masks = self.sam2.track(cv_image)
            #     video_res_masks = video_res_masks.cpu().float() # (num_objects, C, H, W)
            #     video_res_masks = video_res_masks.permute(0, 2, 3, 1) # (num_objects, H, W, C)
            #     video_res_masks = video_res_masks.mean(axis=-1) # (num_objects, H, W)
            #     video_res_masks = (video_res_masks > 0).int()
            #     masks = video_res_masks.numpy().astype(np.uint8)
                
            #     # iterate over masks
            #     for mask in obstacles_bboxes, masks:
            #         mask_uint8 = (mask * 255).astype(np.uint8)
            #         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #         cv2.drawContours(cv_image, contours, -1, (255, 0, 0), 2)

        annotated_frame_msg = self.cv_bridge.cv2_to_imgmsg(cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
        self.annotated_frame_pub.publish(annotated_frame_msg)

def main(args=None):
    rclpy.init(args=args)
    scene_node = ScenePerception()
    rclpy.spin(scene_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()