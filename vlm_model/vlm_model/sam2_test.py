# -*- coding: utf-8 -*-

# AI model related
import torch
from PIL import Image as PILImage
from transformers import pipeline
from pathlib import Path
from sam2.build_sam import build_sam2_camera_predictor

# helper libraries
import cv2
import numpy as np


if __name__ == '__main__':

    first = True
    grounding_dino = pipeline(model="IDEA-Research/grounding-dino-tiny", 
                            task="zero-shot-object-detection", 
                            device="cuda") 
    

    sam2 = build_sam2_camera_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",
                                    "/home/nisarganc/segment_anything/checkpoints/sam2.1_hiera_small.pt")
    
    cap = cv2.VideoCapture(0)
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            ret, cv_image = cap.read()
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            gd_image = PILImage.fromarray(rgb_image)

            if first:
                first = False
                dino_output = grounding_dino(gd_image,  candidate_labels=["objects."], threshold=0.3)
                # [{'score': 0.74167400598526, 'label': 'objects.', 'box': {'xmin': 644, 'ymin': 570, 'xmax': 1122, 'ymax': 1033}}, 
                # {'score': 0.5053098797798157, 'label': 'objects.', 'box': {'xmin': 1772, 'ymin': 909, 'xmax': 1884, 'ymax': 1047}}, 
                # {'score': 0.3090237081050873, 'label': 'objects.', 'box': {'xmin': 808, 'ymin': 734, 'xmax': 959, 'ymax': 891}}]

                bboxes_list = []
                for output in dino_output:
                    box = output['box']
                    bbox = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                    bboxes_list.append(bbox)
                bboxes = np.array(bboxes_list) 
                
                sam2.load_first_frame(rgb_image)
                for i, bbox in enumerate(bboxes):
                    _, out_obj_ids, video_res_masks = sam2.add_new_prompt(frame_idx=0, 
                                                                        obj_id=i,
                                                                        bbox=bbox)
            else:
                out_obj_ids, video_res_masks = sam2.track(rgb_image)
                video_res_masks = video_res_masks.cpu().float() # (num_objects, C, H, W)
                video_res_masks = video_res_masks.permute(0, 2, 3, 1) # (num_objects, H, W, C)
                video_res_masks = video_res_masks.mean(axis=-1) # (num_objects, H, W)
                video_res_masks = (video_res_masks > 0).int()
                masks = video_res_masks.numpy().astype(np.uint8)
                
                # iterate over bboxes and masks
                for i, (bbox, mask) in enumerate(zip(bboxes, masks)):

                    # rgb_image = cv2.rectangle(rgb_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) 

                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgb_image, contours, -1, (255, 0, 0), 2)

                # show image
                res_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("frame", res_image)
                cv2.waitKey(1)