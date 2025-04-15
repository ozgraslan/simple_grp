# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import imageio

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from transformers import AutoConfig, AutoProcessor, AutoBackbone
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def tensor_to_numpy(tensor):
    return tensor.mul(255).clamp_(0, 255).permute(1,2,0).to("cpu", torch.uint8).numpy()

def tensor_to_numpy2(array):
    array -= array.min()
    array /= (array.max() + 1e-8)  # prevent divide by zero
    array *= 255
    return array.astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='base', type=str,
        choices=['small', 'base', 'large', "giant"], help='Architecture (support only ViT atm).')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    vis_config = AutoConfig.from_pretrained("facebook/dinov2-with-registers-giant")
    vis_backbone = AutoBackbone.from_pretrained("facebook/dinov2-with-registers-giant", out_features=[]).to(device)
    vis_processor = AutoProcessor.from_pretrained("facebook/dinov2-with-registers-giant", out_features=[])
    for p in vis_backbone.parameters():
        p.requires_grad = False
    vis_backbone.eval()
    vis_backbone.to(device)


    train_dataset = LeRobotDataset("lerobot/libero_10_image", episodes=[0]) # episodes=train_episodes, 
    print(f"Number of frames in training dataset (100% subset): {len(train_dataset)}")


    train_dl = DataLoader(train_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=device!="cpu",
                          drop_last=False,
                         )

    patch_size = vis_config.patch_size
    num_registers = vis_config.num_register_tokens 
    print(num_registers)

    save_list = [ [[] for _ in range(vis_config.num_attention_heads)] for _ in range(num_registers+2)]
    print(save_list, len(save_list))
    for data in train_dl:
        img = tensor_to_numpy(data["observation.images.image"][0])
        img = vis_processor(img, return_tensors="pt")
        w_featmap = img.pixel_values.shape[-2] // patch_size
        h_featmap = img.pixel_values.shape[-1] // patch_size

        output = vis_backbone(**img.to(device), output_attentions=True, output_hidden_states=True, return_dict=True)

        nh = vis_config.num_attention_heads # number of head
            # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        save_list[-1].append(tensor_to_numpy(torchvision.utils.make_grid(img.pixel_values, normalize=True, scale_each=True)))

        # we keep only the output patch attention
        for reg_idx in range(0, num_registers+1):
            attentions = output.attentions[-1].clone()

            attentions = attentions[0, :, reg_idx, 1+num_registers:].reshape(nh, -1)

            if args.threshold is not None:
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    
            for j in range(nh):
                save_list[reg_idx][j].append(tensor_to_numpy2(attentions[j]))
                # fname = os.path.join(args.output_dir, "reg" + str(reg_idx) + "_attn-head" + str(j) + ".png")
                # plt.imsave(fname=fname, arr=attentions[j], format='png')
                # print(f"{fname} saved.")

            # mean_attentions = attentions.mean(axis=0, keepdims=True)
            # save_list[reg_idx].append(tensor_to_numpy2(mean_attentions[0]))
            # fname = os.path.join(args.output_dir, "reg" + str(reg_idx) + "_attn-head" + "_mean" + ".png")
            # print(mean_attentions.dtype, mean_attentions.shape)
            # exit(0)
            # plt.imsave(fname=fname, arr=mean_attentions[0], format='png')
            # print(f"{fname} saved.")
            
            # max_attentions = attentions.max(axis=0, keepdims=True)
            # save_list[reg_idx].append(tensor_to_numpy2(max_attentions[0]))

            # fname = os.path.join(args.output_dir, "reg" + str(reg_idx) + "_attn-head" + "_max" + ".png")
            # plt.imsave(fname=fname, arr=max_attentions[0], format='png')
            # print(f"{fname} saved.")

            if args.threshold is not None:
                image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
                for j in range(nh):
                    display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "reg" + str(reg_idx) + "_mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)

    for idx in range(0, num_registers+2):
        if idx == 5:
            fname = os.path.join(args.output_dir, "trajectory.gif")
            break

        for j in range(vis_config.num_attention_heads):
            fname = os.path.join(args.output_dir, "reg" + str(idx) + "_attn-head" + str(j) + ".gif")
            imageio.mimsave(fname, save_list[idx][j])
