# import argparse
# import json
import logging
import os
# import pathlib
# import pickle
# import sys
# from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
# import tqdm
# import webdataset
# from ocl import visualizations
from ocl.cli import train
from omegaconf import OmegaConf
from PIL import Image
# from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import (AutoTokenizer, CLIPTextModel, AutoImageProcessor)

# import handlers
# from llm2vec import LLM2Vec
import textwrap


def visualize(prompts, images, outputs, save_path):
    # Prepare a figure for visualization with larger size and smaller gaps between subplots
    fig, axes = plt.subplots(len(prompts), 8, figsize=(24, len(prompts) * 5), squeeze=False)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)  # Reduce horizontal and vertical spacing for a tighter fit

    # Colors for different masks to visually differentiate them
    colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (1, 1, 0),  # Yellow
        (1, 0, 1),  # Magenta
        (0, 1, 1),  # Cyan
        (0.75, 0.75, 0.75),  # Bright Gray
    ]

    # Visualize for each prompt
    for i, prompt in enumerate(prompts):
        # Original image (resized for visualization)
        image_np = images[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(cv2.resize(image_np, (768, 768), interpolation=cv2.INTER_NEAREST))
        axes[i, 0].axis('off')

        # Get masks for the current prompt
        
        image_shape = images[i:i + 1].shape[2:]
        masks_as_image = outputs['object_decoder'].masks_as_image[i]
        masks = masks_as_image.view(-1, 1, *image_shape)
        
        # For each mask, visualize it superimposed on the original image
        for j in range(masks.shape[0]):
            # if j > 4:
            #     break
            # Superimpose the mask on the image
            curr_mask = masks[j].squeeze(0).cpu().detach().numpy()
            
            # Create a colored mask (using different colors for each mask)
            colored_mask = np.zeros((*curr_mask.shape, 3), dtype=np.float32)
            colored_mask[:, :, 0] = curr_mask * colors[j % len(colors)][0] * 1.5  # Amplify intensity by multiplying
            colored_mask[:, :, 1] = curr_mask * colors[j % len(colors)][1] * 1.5  # Amplify intensity by multiplying
            colored_mask[:, :, 2] = curr_mask * colors[j % len(colors)][2] * 1.5  # Amplify intensity by multiplying
            
            # Clip values to be in range [0, 1]
            colored_mask = np.clip(colored_mask, 0, 1)
            
            # Blend the image and the colored mask with a higher emphasis on the mask
            alpha = 0.45  # Increase this value to make the mask more prominent
            blended = image_np * (1 - alpha) + colored_mask * alpha
            
            # Resize to be larger (768x768)
            blended = cv2.resize(blended, (768, 768), interpolation=cv2.INTER_NEAREST)
            wrapped_text = textwrap.fill(prompt[j], width=20)

            # Plot the blended image
            axes[i, j + 1].imshow(blended)
            axes[i, j + 1].set_title(wrapped_text, fontsize=19, pad=7.5, fontweight='bold')  # Add padding to improve readability
            axes[i, j + 1].axis('off')

    # Add a border around subplots for better separation
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

    # Save the combined visualization
    combined_image_path = os.path.join(".", save_path)
    try:
        plt.savefig(combined_image_path, bbox_inches='tight', dpi=300, pad_inches=0, transparent=False)  # Adjust dpi to balance quality and file size
        logging.info(f"Combined visualization saved to {combined_image_path}")
    except Exception as e:
        logging.error(f"Error saving visualization: {e}")
        raise

    print(f"Combined visualization saved to {combined_image_path}")

logging.getLogger().setLevel(logging.INFO)

# TODO: Use CVPR submission checkpoints --- these checkpoints are recent I suppose
CHECKPOINTS = {
    "checkpoint": "/home/mila/o/ozgur.aslan/git/simple_grp/ctrlo/model.ckpt",
    "config": "/home/mila/o/ozgur.aslan/git/simple_grp/ctrlo/config.yaml",
    "text_model": "openai/clip-vit-base-patch32",
}


class CTRLOFeatureExtractor:
    """Handles feature extraction for multiple vision models."""

    def __init__(self):
        self._init_models()
        self._init_transforms()
        
    def _init_models(self):
        # Initialize Ctrlo
        config_path = CHECKPOINTS[
            "config"
        ]
        encoder_checkpoint_path = CHECKPOINTS[
            "checkpoint"
        ]
        oclf_config = OmegaConf.load(config_path)
        self.ctrlo_model = train.build_model_from_config(
            oclf_config, encoder_checkpoint_path
        )
        text_model_path = CHECKPOINTS[
            "text_model"
        ]
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = CLIPTextModel.from_pretrained(text_model_path)
        self.bbox_cent = torch.tensor([[-1, -1]] * 7, dtype=torch.float32)
        self.bbox_inst = torch.tensor([[-1, -1, -1, -1]] * 7, dtype=torch.float32)


    def _init_transforms(self):        
        img_size = 224
        self.img_transform = transforms.Compose(
            [   
                transforms.Lambda(lambda img: img if isinstance(img, torch.Tensor) else torch.from_numpy(img) if isinstance(img, np.ndarray) else torch.from_numpy(np.array(img))),
                transforms.Lambda(lambda img: img.permute(0,3,1,2) if len(img.shape) == 4 else img.permute(2,0,1)),
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(
                    img_size,
                ),

                transforms.Lambda(lambda img: img.div(255) if isinstance(img, torch.ByteTensor) else img),
                transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
                transforms.Lambda(lambda img: img.unsqueeze(0) if len(img.shape) == 3 else img),
            ]
                )

    def to(self, device):
        self.ctrlo_model = self.ctrlo_model.to(device=device)
        self.text_model = self.text_model.to(device=device)
        self.bbox_cent = self.bbox_cent.to(device=device)
        self.bbox_inst = self.bbox_inst.to(device=device)
        return self

    @property
    def device(self):
        return self.bbox_cent.device

    def eval(self):
        self.ctrlo_model.eval()
        self.text_model.eval()

    def train(self):
        self.ctrlo_model.train()
        self.text_model.train()
        
    def embed_text(self, text):
        tokenized_text = self.text_tokenizer(text, 
                                             padding="max_length", max_length=32,
                                             return_tensors="pt").input_ids.to(self.text_model.device)
        embeded_text = self.text_model(input_ids=tokenized_text).pooler_output
        contrastive_loss_mask = torch.tensor([int(p != "other") for p in text])
        return embeded_text, contrastive_loss_mask

    
    def embed_img_single_text(self, img, text):
        transformed_img = self.img_transform(img)
        embeded_text, contrastive_loss_mask = self.embed_text(text)
        print(transformed_img.shape, embeded_text.shape)
        outputs = self.extract_features_batch(transformed_img, contrastive_loss_mask.unsqueeze(0), embeded_text.unsqueeze(0))
        return outputs

    def embed_batch_img(self, img):
        transformed_img = self.img_transform(img)
        bsz = transformed_img.shape[0]
        print(transformed_img.shape)
        empty_loss_mask = torch.zeros((bsz, 7), dtype=int, device=transformed_img.device)
        embeded_empty_text = torch.ones((bsz,7,512), device=transformed_img.device)
        outputs = self.extract_features_batch(transformed_img, empty_loss_mask, embeded_empty_text)
        return outputs

    def embed_batch_img_single_text(self, img_batch, text):
        transformed_img = self.img_transform(img_batch)
        bsz = transformed_img.shape[0]
        embeded_text, contrastive_loss_mask = self.embed_text(text)
        outputs = self.extract_features_batch(transformed_img, contrastive_loss_mask.repeat(bsz, 1), embeded_text.repeat(bsz, 1, 1))
        return outputs

    def extract_features_batch(self, images, contrastive_loss_mask, name_embeddings):
        """Extract features for a batch of images."""
        bsz = images.shape[0]
        # print(images.dtype, self.bbox_inst.dtype)
        # print(images.shape, contrastive_loss_mask.shape, name_embeddings.shape)
        inputs = {
            "image": images.to(device=self.bbox_inst.device),
            "bbox_centroids": self.bbox_cent.repeat(bsz, 1, 1),
            "contrastive_loss_mask": contrastive_loss_mask.to(device=self.bbox_inst.device),
            "name_embedding": name_embeddings.to(device=self.bbox_inst.device),
            "instance_bbox": self.bbox_inst.repeat(bsz, 1, 1),
            "batch_size": bsz,
        }
        outputs = self.ctrlo_model(inputs)
        return outputs


def denormalize(image_tensor, mean, std):
    # Ensure the tensor is in the right format
    image_tensor = image_tensor * torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1) + torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
    return image_tensor

if __name__ == "__main__":
    # "put both the alphabet soup and the cream cheese box in the basket"
    # you can specify upto 7 regions or objects phrases, the rest will be "other"
    prompts = [
        ["the alphabet soup", "the blue cream cheese box", "the basket box", "manipulator", "the orange juice box", "other", "other"],
        ["left dog", "right dog", "grass", "flowers", "eyes", "ears", "other"],

    ]
    images = [
        "../data/agentview_image.png",
        "../data/agentview_image.png",
    ]

    feature_extractor = CTRLOFeatureExtractor().to(device="cuda")
    np_images = np.stack([np.array(Image.open(img).convert("RGB")) for img in images], axis=0)
    save_path = "img_only_224_cc.png"
    outputs = feature_extractor.embed_batch_img(np_images[0]) #, , prompts[0]
    print(outputs["feature_extractor"].features.shape)
    denormed_img = denormalize(feature_extractor.img_transform(np_images), [0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    visualize([[""]*7], denormed_img, outputs, save_path)
    # print(outputs.keys())
    pg = outputs["perceptual_grouping"]
    print(pg.objects, pg.feature_attributions.shape)
    exit(0)