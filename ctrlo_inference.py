import logging
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
from ocl.cli import train
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoImageProcessor

from llm2vec import LLM2Vec
import textwrap

from utils import is_uint8

logging.getLogger().setLevel(logging.INFO)

CHECKPOINTS = {
    "checkpoint": "/network/scratch/o/ozgur.aslan/ctrlo/pretrained_model.ckpt",
    "config": "/network/scratch/o/ozgur.aslan/ctrlo/config.yaml",
    "text_model": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"),
}

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
    if images.dtype == np.uint8:
        images = images / 255.0 
    # Visualize for each prompt
    for i, prompt in enumerate(prompts):
        # Original image (resized for visualization)
        image_np = images[i].copy()
        axes[i, 0].imshow(cv2.resize(image_np, (768, 768), interpolation=cv2.INTER_NEAREST))
        axes[i, 0].axis('off')

        # Get masks for the current prompt
        
        image_shape = image_np.shape[:-1]
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


class CTRLOFeatureExtractor(nn.Module):
    """Handles feature extraction for multiple vision models."""

    def __init__(self, config=None):
        super().__init__()
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
        text_model_tuple = CHECKPOINTS[
            "text_model"
        ]
        self.text_model = LLM2Vec.from_pretrained(
            text_model_tuple[0],
            peft_model_name_or_path=text_model_tuple[1],
            torch_dtype=torch.bfloat16
        )
        self.bbox_cent = nn.Parameter(torch.tensor([[-1, -1]] * 7, dtype=torch.float32)) 
        self.bbox_inst = nn.Parameter(torch.tensor([[-1, -1, -1, -1]] * 7, dtype=torch.float32))

    def _init_transforms(self):        
        processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov2-with-registers-small', 
            size={"height": 224, "width": 224}, 
            do_center_crop=False,
            do_resize=True,
        )
        print(processor)
        self.img_transform = lambda imgs, do_rescale: processor(images=imgs, do_rescale=do_rescale, return_tensors="pt").pixel_values

    # def to(self, device_or_dtype):
    #     self.ctrlo_model = self.ctrlo_model.to(device_or_dtype)
    #     self.text_model = self.text_model.to(device_or_dtype)
    #     self.bbox_cent = self.bbox_cent.to(device_or_dtype)
    #     self.bbox_inst = self.bbox_inst.to(device_or_dtype)
    #     return self

    # @property
    # def device(self):
    #     return self.bbox_cent.device

    # def eval(self):
    #     self.ctrlo_model.eval()
    #     self.text_model.eval()

    # def train(self):
    #     self.ctrlo_model.train()
    #     self.text_model.train()

    def extract_features_batch(self, images, contrastive_loss_mask, name_embeddings):
        """Extract features for a batch of images."""
        bsz = images.shape[0]
        inputs = {
            "image": images.to(device=self.device),
            "bbox_centroids": self.bbox_cent.repeat(bsz, 1, 1),
            "contrastive_loss_mask": contrastive_loss_mask.to(device=self.device),
            "name_embedding": name_embeddings.to(device=self.device),
            "instance_bbox": self.bbox_inst.repeat(bsz, 1, 1),
            "batch_size": bsz,
        }
        outputs = self.ctrlo_model(inputs)
        return outputs

    def forward(self, images, text_masks, text_embeds):
        transformed_img = self.img_transform(images, do_rescale = is_uint8(images))
        outputs = self.extract_features_batch(transformed_img, text_masks, text_embeds)
        return outputs

    @torch.no_grad()
    def embed_text(self, text_list):
        embeded_text = self.text_model.encode(text_list)
        contrastive_loss_mask = torch.tensor([int(text != "other") for text in text_list])
        return embeded_text, contrastive_loss_mask
    
    @torch.no_grad()
    def embed_img_text(self, img, text_embed, text_mask):
        return self.forward(img, text_mask.unsqueeze(0), text_embed.unsqueeze(0))

    @torch.no_grad()
    def embed(self, images, task_index, text_embeds, text_masks):
        outputs = self.forward(images, text_masks[task_index], text_embeds[task_index])
        return  {"patch": outputs["feature_extractor"].features, 
                 "slot": outputs["perceptual_grouping"].objects}


    @torch.no_grad()
    def prepare_tasks(self, tasks, task2obj, **kwargs):
        text_embed_list = []
        text_mask_list = []
        task_ids = []

        ## did a loop because I am not sure if items returned in order
        for task_id, task_text in tasks.items():
            obj_text_list = task2obj[task_text]
            task_ids.append(task_id)
            text_embed, text_mask = self.embed_text(obj_text_list)
            # print(text_embed.shape, text_mask.shape)
            text_embed_list.append(text_embed)
            text_mask_list.append(text_mask)

        task_ids = torch.tensor(task_ids, dtype=torch.long)

        ## preserve order using task ids
        text_embeds = torch.stack(text_embed_list, dim=0)[task_ids]
        text_masks = torch.stack(text_mask_list, dim=0)[task_ids]
        return {"text_embeds": text_embeds, "text_masks": text_masks}



if __name__ == "__main__":
    # "put both the alphabet soup and the cream cheese box in the basket"
    # you can specify upto 7 regions or objects phrases, the rest will be "other"
    prompt = ["The robot arm", "the alphabet soup", "the cream cheese box", "the basket box", "the orange juice box", "other", "other"]

    image = "./data/agentview_image.png"

    feature_extractor = CTRLOFeatureExtractor().to("cuda")
    np_images = np.array(Image.open(image).convert("RGB").resize((224, 224)), dtype=np.uint8)
    save_path = "test4.png"

    embeded_text, contrastive_loss_mask = feature_extractor.embed_text(prompt)
    outputs = feature_extractor.embed_img_text(np_images, embeded_text, contrastive_loss_mask)
    print(outputs.keys())
    print(outputs["feature_extractor"].features.shape)
    visualize([prompt], np_images[None], outputs, save_path)
