import numpy as np
import open3d as o3d
import json
import nicegui
import cv2
from PIL import Image
import asyncio


from utils import vis_img_3d, points_camera2image, bbox3d2corners_camera, read_calib, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, bbox3d2corners

@nicegui.ui.page('/viewer')
def viewer():

    def update_scene(scene, pc = [], bboxes=[]):
        scene.clear()
        if not(isinstance(pc, list)):
            points = pc.tolist()
        else:
            points = pc
        colors = np.zeros([len(pc),3])
        colors[:,0] = 255
        point_cloud = scene.point_cloud(points, colors, point_size=0.03)

        lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        for i in range(len(bboxes)):
            for k in range(len(lines)):
                scene.line(bboxes[i][lines[k][0]], bboxes[i][lines[k][1]]).material('#0000FF')
        print('done')


    def generate_data(index, pc, bboxes_pt, bboxes_trt):
        index = f"{index:06d}"
        json_loc = "json_output/"+index+"_trtfp16.json"
        json_pytorch_loc = "json_output_pytorch/"+index+".json"
        pointcloud_loc = "/toshiba_1tb/home/Pointpillar_data/kitti/testing/velodyne/"+index+".bin"

        img_loc = "/toshiba_1tb/home/Pointpillar_data/kitti/testing/image_2/"+index+".png"
        
        img = cv2.imread(img_loc)
        imgpt = cv2.imread(img_loc)
        image_shape = img.shape[:2]

        calib_loc = "/toshiba_1tb/home/Pointpillar_data/kitti/testing/calib/"+index+".txt"
        calib_info = read_calib(calib_loc)
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)
        pc_data=  np.fromfile(pointcloud_loc, dtype=np.float32).reshape(-1, 4)[:,:3]
        pc.clear()
        pc.extend(pc_data.tolist())

        ### pre process trt json
        f = open(json_loc)
        data_trt = json.load(f)
        #print(json.dumps(data, indent =4))

        lidar_bboxes = []
        scores = []
        labels = []
        images.clear()
        for i, bbox_json in enumerate(data_trt['bounding boxes']):
            if not isinstance(bbox_json, dict):
                print(f"⚠️ Warning: entry {i} is not a dict:", bbox_json)
                continue
            lidar_bboxes.append([bbox_json['x'],bbox_json['y'],bbox_json['z'],
                            bbox_json['w'],bbox_json['l'],bbox_json['h'],
                            bbox_json['theta']])
            scores.append(bbox_json['score'])
            labels.append(bbox_json['label'])

        results_trt = {
            'lidar_bboxes' : np.array(lidar_bboxes),
            'scores' : np.array(scores),
            'labels' : np.array(labels),
            "avg_model_time": np.array(data_trt["avg_model_time"]),
            "avg_post_time": np.array(data_trt["avg_post_time"]),
            "avg_pre_time": np.array(data_trt["avg_pre_time"]),
            "avg_total_time": np.array(data_trt["avg_total_time"])

        }
        trt_data.clear()
        trt_data.append(results_trt)


        results_trt_filter_img = keep_bbox_from_image_range(results_trt, tr_velo_to_cam, r0_rect, P2, image_shape)
        results_trt_filter_3d = keep_bbox_from_lidar_range(results_trt, pcd_limit_range)

        labels, scores = results_trt_filter_img['labels'], results_trt_filter_img['scores']
        bboxes2d, camera_bboxes = results_trt_filter_img['bboxes2d'], results_trt_filter_img['camera_bboxes'] 
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        image_points = points_camera2image(bboxes_corners, P2)
        temp = vis_img_3d(img, image_points, labels, rt=True)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = Image.fromarray(temp)
        images.append(temp)

        lidar_bboxes_trt = results_trt_filter_3d['lidar_bboxes']
        bboxes_trt_data = bbox3d2corners(lidar_bboxes_trt)

        bboxes_trt.clear()
        bboxes_trt.extend(bboxes_trt_data.tolist())

        ###pre process pytorch json

        j = open(json_pytorch_loc)
        data_pt = json.load(j)

        results_pt = {
            'lidar_bboxes' : np.array(data_pt['lidar_bboxes']),
            'scores' : np.array(data_pt['scores']),
            'labels' : np.array(data_pt['labels']),
            "avg_model_time": np.array(data_pt["avg_model_time"]),
            "avg_post_time": np.array(data_pt["avg_post_time"]),
            "avg_pre_time": np.array(data_pt["avg_pre_time"]),
            "avg_total_time": np.array(data_pt["avg_total_time"])
        }
        pt_data.clear()
        pt_data.append(results_pt)

        results_pt_filter_img = keep_bbox_from_image_range(results_pt, tr_velo_to_cam, r0_rect, P2, image_shape)
        results_pt_filter_3d = keep_bbox_from_lidar_range(results_pt, pcd_limit_range)

        labels_pt, scores_pt = results_pt_filter_img['labels'], results_pt_filter_img['scores']
        bboxes2d, camera_bboxes_pt = results_pt_filter_img['bboxes2d'], results_pt_filter_img['camera_bboxes'] 
        bboxes_corners_pt = bbox3d2corners_camera(camera_bboxes_pt)
        image_points_pt = points_camera2image(bboxes_corners_pt, P2)
        temp1 = vis_img_3d(imgpt, image_points_pt, labels_pt, rt=True)
        temp1 = cv2.cvtColor(temp1, cv2.COLOR_BGR2RGB)
        temp1 = Image.fromarray(temp1)
        images.append(temp1)

        lidar_boxes_pt = results_pt_filter_3d['lidar_bboxes']
        bboxes_pt_data = bbox3d2corners(lidar_boxes_pt)

        bboxes_pt.clear()
        bboxes_pt.extend(bboxes_pt_data.tolist())

        return pc, bboxes_pt, bboxes_trt


    def check_input(i):
        try:
            n = int(i)
        except:
            print("not an int")
            return
        
    def update_card(card, data):
        with card:
            with nicegui.ui.grid(columns = 10):
                nicegui.ui.label("No.")
                nicegui.ui.label("X")
                nicegui.ui.label("Y")
                nicegui.ui.label("Z")
                nicegui.ui.label("W")
                nicegui.ui.label("L")
                nicegui.ui.label("H")
                nicegui.ui.label("theta")
                nicegui.ui.label("Score")
                nicegui.ui.label("Label")
                for i, boxes in enumerate(data['lidar_bboxes']):
                    nicegui.ui.label(str(i+1))
                    nicegui.ui.label("{:.5f}".format(boxes[0]))
                    nicegui.ui.label("{:.5f}".format(boxes[1]))
                    nicegui.ui.label("{:.5f}".format(boxes[2]))
                    nicegui.ui.label("{:.5f}".format(boxes[3]))
                    nicegui.ui.label("{:.5f}".format(boxes[4]))
                    nicegui.ui.label("{:.5f}".format(boxes[5]))
                    nicegui.ui.label("{:.5f}".format(boxes[6]))
                    nicegui.ui.label("{:.5f}".format(data["scores"][i]))
                    nicegui.ui.label("{:.5f}".format(data["labels"][i]))
            nicegui.ui.label("Average pre-processing time: " + str(data["avg_pre_time"]))
            nicegui.ui.label("Average inference time: " + str(data["avg_model_time"]))
            nicegui.ui.label("Average post-processing time: " + str(data["avg_post_time"]))
            nicegui.ui.label("Average total time: " + str(data["avg_total_time"]))



        
    def generate_and_update():
        print('generating')
        try:
            index = int(input.value)
            check_input(index)

            print("generating...")
            pc.clear()
            bboxes_pt.clear()
            bboxes_trt.clear()

            generate_data(index, pc, bboxes_pt, bboxes_trt)

            cardtrt.clear()
            update_card(cardtrt, trt_data[0])
            cardpt.clear()
            #print(pt_data)
            update_card(cardpt, pt_data[0])

            imagetrt.set_source(images[0])
            imagetrt.update
            imagept.set_source(images[1])
            imagept.update


            update_scene(scenetrt, pc, bboxes_trt)
            update_scene(scenept, pc, bboxes_pt)
            #for i, data in enumerate(trt_data['lidar_bboxes']):


        except Exception as e:
            print(f"Error: {e}")



    with  nicegui.ui.grid(columns = 2):
        pc = []
        bboxes_pt = []
        bboxes_trt = []
        trt_data = []
        pt_data = []
        images = []

        input = nicegui.ui.input(label = 'which index to compare', placeholder= '0-439')
        nicegui.ui.button('generate', on_click=lambda: [generate_and_update()])

        nicegui.ui.label("TensorRT FP16")
        nicegui.ui.label("PyTorch")

        cardtrt = nicegui.ui.card()
        cardpt = nicegui.ui.card()

        imagetrt = nicegui.ui.image()
        imagept = nicegui.ui.image()

        with nicegui.ui.scene(width = 1024, height =512, grid = False) as scenetrt:
            update_scene(scenetrt)
        with nicegui.ui.scene(width = 1024, height =512, grid = False) as scenept:
            update_scene(scenept)

pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
fp16 = True
nicegui.ui.run()
nicegui.ui.navigate.to('/viewer')