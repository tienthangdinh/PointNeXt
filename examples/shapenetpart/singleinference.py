import numpy as np
import open3d as o3d
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from openpoints.models import build_model_from_cfg
from openpoints.utils.config import EasyConfig
import matplotlib.pyplot as plt

# === Step 1: Load Configuration and Pretrained Model ===
cfg_path = "cfgs/shapenetpart/pointnext-s.yaml"
pretrained_path = "checkpoint/shapenetpart-train-pointnext-s-ngpus4-seed5011-20220821-170334-J6Ez964eYwHHPZP4xNGcT9_ckpt_best.pth"

cfg = EasyConfig()
cfg.load(cfg_path, recursive=True)
model = build_model_from_cfg(cfg.model).cuda()
model.load_state_dict(torch.load(pretrained_path)['model'])
model.eval()

cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                   'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
                   'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
                   'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}

# === Step 2: Load and Process Input Point Cloud ===
def load_point_cloud(filename, num_points=2048):
    pcd = o3d.io.read_point_cloud(filename)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))  # Estimate normals
    
    points = np.asarray(pcd.points)  # XYZ coordinates
    normals = np.asarray(pcd.normals)  # Estimated normals

    # Handle color data if present
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)  # RGB values (normalized [0,1])
    else:
        colors = np.zeros((len(points), 3))  # If no color, fill with zeros

    # Compute height feature (relative to min height)
    heights = points[:, 1] - np.min(points[:, 1])
    heights = heights.reshape(-1, 1)  # Shape (N, 1)

    # Ensure we have exactly 2048 points (downsample/upsample)
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
    else:
        idx = np.random.choice(len(points), num_points, replace=True)

    points, normals, heights = points[idx], normals[idx], heights[idx]

    # Combine all features: [XYZ, Normals, Heights]
    input_data = np.hstack((points, normals, heights))  # Shape: (2048, 7)
    return input_data

# === Step 3: Define a Fixed Set of Colors (Max 5 Colors) ===
def generate_fixed_colors():
    return np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255]   # Purple
    ]) / 255.0  # Normalize to range [0, 1] for Open3D

FIXED_COLORS = generate_fixed_colors()

SHAPENET_CLASSES = {
    "airplane": 0, "bag": 1, "cap": 2, "car": 3, "chair": 4,
    "earphone": 5, "guitar": 6, "knife": 7, "lamp": 8, "laptop": 9,
    "motorbike": 10, "mug": 11, "pistol": 12, "rocket": 13, "skateboard": 14, "table": 15
}
# === Step 4: Run Model Inference ===
def run_inference(filename):
    input_data = load_point_cloud(filename)

    # Convert to Tensor & Format Correctly
    pos_tensor = torch.tensor(input_data[:, :3], dtype=torch.float32).unsqueeze(0).cuda()  # (1, 2048, 3)
    normals_tensor = torch.tensor(input_data[:, 3:6], dtype=torch.float32).unsqueeze(0).cuda()  # (1, 2048, 3)
    heights_tensor = torch.tensor(input_data[:, 6:], dtype=torch.float32).unsqueeze(0).cuda()  # (1, 2048, 1)

    # Construct Feature Tensor (Shape: (1, 7, 2048))
    x_tensor = torch.cat([pos_tensor, normals_tensor, heights_tensor], dim=2).permute(0, 2, 1)

    # Dummy class tensor (we assume object belongs to class 0)
    cls_tensor = torch.tensor([[16]], dtype=torch.int64).cuda()

    input_dict = {"pos": pos_tensor, "x": x_tensor, "cls": cls_tensor}

    print(f"🔍 Model Input Shapes:")
    for key, value in input_dict.items():
        print(f"{key}: {value.shape}")

    with torch.no_grad():
        logits = model(input_dict)
        preds = torch.argmax(logits, dim=1).cpu().numpy().squeeze()  # Shape: (2048,)

    return input_data[:, :3], preds  # Return points and predictions

# === Step 5: Visualization Function ===
def visualize_prediction(points, preds):
    unique_labels = np.unique(preds)  # Get unique predicted labels
    label_to_color = {label: FIXED_COLORS[i % len(FIXED_COLORS)] for i, label in enumerate(unique_labels)}

    pred_colors = np.array([label_to_color[label] for label in preds])  # Assign color per label

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pred_colors)

    o3d.visualization.draw_geometries([pcd])


# === Step 6: Run on an Example File ===
pc_file = "/home/thang/Desktop/work/machine_segmentation/decom3scans/data/pose1_0.ply"  # Change this to your point cloud file
points, preds = run_inference(pc_file)
visualize_prediction(points, preds)
