import argparse
import yaml
import os
import sys
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.utils.config import EasyConfig
from vis_shapenet import vis_shapenet

# ===> Load Configuration
cfg_path = "cfgs/shapenetpart/pointnext-s.yaml"
pretrained_path = "checkpoint/shapenetpart-train-pointnext-s-ngpus4-seed5011-20220821-170334-J6Ez964eYwHHPZP4xNGcT9_ckpt_best.pth"

cfg = EasyConfig()
cfg.load(cfg_path, recursive=True)

# ===> Build Model
model = build_model_from_cfg(cfg.model).cuda()
model.load_state_dict(torch.load(pretrained_path)['model'])
model.eval()

# ===> Load Dataset in Correct Format
val_loader = build_dataloader_from_cfg(
    batch_size=1,  # Process one sample at a time
    dataset_cfg=cfg.dataset,
    dataloader_cfg=cfg.dataloader,
    datatransforms_cfg=cfg.datatransforms,
    split='val',
    distributed=False
)

# ===> Run Inference
for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
    print(data)
    # Move data to CUDA
    for key in data.keys():
        data[key] = data[key].cuda(non_blocking=True)

    # Run inference
    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1).cpu().numpy().squeeze()

    # Extract points
    points = data["pos"].cpu().numpy().squeeze()

    # Visualize
    vis_shapenet(data, preds, idx)
    
    break  # Only process one example
