
import torch
import numpy as np
from PIL import Image
from typing import Any, List, Dict, Optional, Tuple
from transformers import pipeline, AutoModelForMaskGeneration, AutoProcessor
from vlm_model.Utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float,
    detector_id: str
) -> List[Dict[str, Any]]:

    object_detector = pipeline(model=detector_id, 
                               task="zero-shot-object-detection", device=device)

    labels = [label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    # remove aruco bboxes from results here
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool,
    segmenter_id: str
) -> List[DetectionResult]:

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results) # replace with object marker id corners
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Image.Image,
    labels: List[str],
    threshold: float,
    polygon_refinement: bool,
    detector_id: str,
    segmenter_id: str
) -> Tuple[np.ndarray, List[DetectionResult]]:

    detections = detect(image, labels, threshold, detector_id)

    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return detections

if __name__ == "__main__":

    image = Image.open("scene_dataset/11_scene_setup_obstacles2.jpg")

    # detect obstacles and segment object
    labels = ["detect objects"]
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    results = grounded_segmentation(image, labels, threshold=0.3, polygon_refinement=False,
                                    detector_id=detector_id, segmenter_id=segmenter_id)

    # plot detections
    plot_detections(image, results, save_name="detection_results.jpg")

    
