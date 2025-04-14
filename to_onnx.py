#import torch.onnx
import torch
import os
from pointpillars.utils import read_points
from pointpillars.model import PointPillars


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

#read point clouds
pc_path = 'C:/Users/uidp10097/Practise Code/Pointpillar_pc/raw_data/velodyne/000005.bin'
pc = read_points(pc_path)
pc = point_range_filter(pc)
pc_torch = torch.from_numpy(pc)
print(pc_torch.shape)

#load model
model = PointPillars().cuda()  # Replace with your model class
model.load_state_dict(torch.load("pretrained/epoch_160.pth"))
model.eval()
print(model)

#test with input
# with torch.no_grad():
#     pc_torch = pc_torch.cuda()
#     result_filter = model(batched_pts=[pc_torch], 
#                               mode='test')[0]
# print(result_filter)


onnx_path = 'Pointpillar_onnx/pointpillars.onnx'
# dummy_input = torch.randn(63339,4)

# torch.onnx.export(model, 
#                   (dummy_input,), 
#                   onnx_path,
#                   input_names=["points", "img_metas"], 
#                   output_names=["bboxes", "labels", "scores"],
#                   opset_version=11,  # Ensure compatibility
#                   dynamic_axes={'points': {0: 'batch_size'}})

print("PointPillars model exported to ONNX:", onnx_path)