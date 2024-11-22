import argparse
import torch
from main import get_args_parser
from models.ULIP_models import ULIP2_PointBERT_Colored
import os
import pickle
import pandas as pd
import numpy as np
import open3d as o3d

from collections import OrderedDict
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    model = ULIP2_PointBERT_Colored(args)
    checkpoint = torch.load('models/ULIPv2.pt', map_location='cuda')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():  # Use `checkpoint` directly if saved without `state_dict`
        name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
        new_state_dict[name] = v
    for k, v in model.open_clip_model.state_dict().items():
        name = "open_clip_model." + k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    colors = np.load("models/points_colors.npy")
    points_array = np.asarray(o3d.io.read_point_cloud("models/scene_example.ply").points)
    print(points_array.shape)
    print(colors.shape)
    points_array = np.concatenate((points_array, colors), axis=1)
    print(points_array.shape)
    points_array = points_array[np.newaxis, ...]
    points_array = torch.from_numpy(points_array).to(device = "cpu", dtype=torch.float)
    print(model.encode_pc(points_array))
