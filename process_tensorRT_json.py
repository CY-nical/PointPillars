import open3d
from tqdm import tqdm
import os
import json
import numpy
import cv2
import numpy as np
from utils import vis_img_3d, points_camera2image, bbox3d2corners_camera, read_calib, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range

index = "000000"
fp16 = True
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

def generate_img(index):
    index = f"{index:06d}"
    if(fp16 == True):
        json_loc = "json_output/"+index+"_trtfp16.json"
    else:
        json_loc = "json_output/"+index+".json"
    img_loc = "/toshiba_1tb/home/Pointpillar_data/kitti/testing/image_2/"+index+".png"
    img = cv2.imread(img_loc)

    calib_loc = "/toshiba_1tb/home/Pointpillar_data/kitti/testing/calib/"+index+".txt"
    calib_info = read_calib(calib_loc)
    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
    r0_rect = calib_info['R0_rect'].astype(np.float32)
    P2 = calib_info['P2'].astype(np.float32)

    f = open(json_loc)


    data = json.load(f)
    #print(json.dumps(data, indent =4))

    lidar_bboxes = []
    scores = []
    labels = []

    for i, bbox_json in enumerate(data['bounding boxes']):
        if not isinstance(bbox_json, dict):
            print(f"⚠️ Warning: entry {i} is not a dict:", bbox_json)
            continue
        lidar_bboxes.append([bbox_json['x'],bbox_json['y'],bbox_json['z'],
                        bbox_json['w'],bbox_json['l'],bbox_json['h'],
                        bbox_json['theta']])
        scores.append(bbox_json['score'])
        labels.append(bbox_json['label'])

    results = {
        'lidar_bboxes' : np.array(lidar_bboxes),
        'scores' : np.array(scores),
        'labels' : np.array(labels)
    }

    print(results['labels'])

    image_shape = img.shape[:2]
    result_filter = keep_bbox_from_image_range(results, tr_velo_to_cam, r0_rect, P2, image_shape)
    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']
    bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
    bboxes_corners = bbox3d2corners_camera(camera_bboxes)
    image_points = points_camera2image(bboxes_corners, P2)
    img = vis_img_3d(img, image_points, labels, rt=True)
    cv2.imwrite("processed_img_trt/"+index+".png", img)


for i in tqdm(range(100)):
    generate_img(i)
    
# print(bounding_boxes.shape)
# bb_corners = bbox3d2corners_camera(bounding_boxes)
# print(bb_corners)

