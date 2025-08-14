"""
Simple GRP
A simple implementation of Octo using PyTorch. 
Based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
"""
import os
# import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# import transformers
from transformers import AutoTokenizer, AutoModel
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

import wandb
import tyro
from tensordict import TensorDict

from dataclasses import dataclass, field
from simple_grp import SimpleGRP, PROJLAYERDICT
from ctrlo_inference import CTRLOFeatureExtractor
from utils import set_seed_everywhere, get_att_mask

DTYPEMAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
}    

@dataclass
class GRPConfig: 
    save_path: str
    dataset_path: str
    hf_user_id: str
    dataset_repo_id: str
    policy_name: str

    seed: int = 0
    depth: int = 6
    num_heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 2048
    hidden_dim: int = 512

    wproject: str = "sgrp-libero10"
    use_wandb: bool = True
    push_to_hf: bool = True

    torch_compile: bool = False
    learning_rate: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 256
    num_action_chunks: int = 32
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype_str: str = "float32"
    log_freq: int = 100

    text_dim: int = 512
    patch_dim: int = 384
    slot_dim: int = 256
    state_dim: int = 8
    action_dim: int = 7

    num_input_tokens: int = 32 + 7 + 256
    num_state_tokens: int = 1 
    
    text_model_name: str = "google-t5/t5-small"
    text_padding: str = "max_length"
    text_max_length: int = 32

    projection_layer: str = "ctrlo_linear_projection"
    action_head: str = "continuous"

    remove_encoders: bool = True

    @property
    def num_tokens(self) -> int:
        return self.num_input_tokens + self.num_state_tokens + self.num_action_chunks
    
    @property
    def policy_save_path(self) -> str:
        return os.path.join(self.save_path, self.policy_name)

    @property
    def hf_repo_id(self) -> str:
        return os.path.join(self.hf_user_id, self.policy_name)
    
    @property
    def dtype(self) -> torch.dtype:
        return DTYPEMAP[self.dtype_str]




class CTRLOLinearProjectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_projection = nn.Linear(config.text_dim, config.hidden_dim)
        self.patch_projection = nn.Linear(config.patch_dim, config.hidden_dim)
        self.slot_projection = nn.Linear(config.slot_dim, config.hidden_dim)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)

    def forward(self, x):
        text_emb  = self.text_projection(x["text"])
        patch_emb = self.patch_projection(x["patch"])
        slot_emb = self.slot_projection(x["slot"])
        state_emb = self.state_projection(x["state"]).unsqueeze(1)
        return torch.cat([text_emb, slot_emb, patch_emb, state_emb], dim=1)

PROJLAYERDICT["ctrlo_linear_projection"] = CTRLOLinearProjectionLayer

class CtrloGRPPolicy(nn.Module):
    def __init__(self, config: GRPConfig):
        super().__init__()
        self.config = config
        self.image_encoder = CTRLOFeatureExtractor()

        self.text_tokenizer =  AutoTokenizer.from_pretrained(
            config.text_model_name,
            torch_dtype=config.dtype
        )
        self.text_encoder = AutoModel.from_pretrained(
            config.text_model_name,
            torch_dtype=config.dtype
        ).encoder
        self.simple_grp = SimpleGRP(config)

        print(
            "CTRLO:", self.image_encoder.training, 
            next(self.image_encoder.parameters()).requires_grad, 
            next(self.image_encoder.ctrlo_model.parameters()).requires_grad,
            next(self.image_encoder.text_model.parameters()).requires_grad
        )
        print("Text Encoder:", self.text_encoder.training, next(self.text_encoder.parameters()).requires_grad)
        print("Simple GRP:", self.simple_grp.training, next(self.simple_grp.parameters()).requires_grad)

        ## precompute causal block attention combined with precomputed text token masks of the tasks
        ## we can precompute this because there are only 10 task descriptions
        block_mask_arr = torch.tensor(
            [1] + (config.num_input_tokens - 1) * [0] + \
            [1] + (config.num_state_tokens - 1) * [0] + \
            [1] + (config.num_action_chunks - 1) * [0]
        )
        ## precompute text_embeddings and text_masks 
        with torch.no_grad():
            text_embeds, text_masks = self.get_task_text_embeds(config.tasks)

        text_masks = torch.cat([text_masks, torch.ones((text_masks.size(0), config.num_tokens-text_masks.size(1)), dtype=text_masks.dtype)], dim=1)
        attention_mask = get_att_mask(
            block_mask_arr.unsqueeze(0).repeat(text_masks.shape[0], 1), 
            text_masks
        )
        attention_mask = torch.logical_not(attention_mask) # this is for nn.MultiHeadAttention
        attention_mask = attention_mask.unsqueeze(1).repeat(1, config.num_heads, 1, 1)

        self.register_buffer("text_embeds", text_embeds)
        self.register_buffer("attention_mask", attention_mask)

        print("text_embeds:", self.text_embeds.size(), self.text_embeds.requires_grad)
        print("attention_mask:", self.attention_mask.size(), self.attention_mask.requires_grad)

        if self.config.remove_encoders:
            del self.image_encoder
            del self.text_encoder
            del self.text_tokenizer
        else:
            self.image_encoder.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.image_encoder.eval()
            self.text_encoder.eval()


    def encode_txt(self, text):
        text_tokens = self.text_tokenizer(
            text, return_tensors="pt", 
            padding=self.config.text_padding, 
            max_length=self.config.text_max_length
        )
        text_embeds = self.text_encoder(input_ids=text_tokens.input_ids).last_hidden_state
        return text_embeds, text_tokens.attention_mask

    def get_task_text_embeds(self, tasks):
        text_embed_list = []
        text_mask_list = []
        task_ids = []

        ## did a loop because I am not sure if items returned in order
        for task_id, task_text in tasks.items():
            task_ids.append(task_id)
            text_embed, text_mask = self.encode_txt(task_text)
            text_embed_list.append(text_embed)
            text_mask_list.append(text_mask)

        task_ids = torch.tensor(task_ids, dtype=torch.long)
        ## preserve order using task ids
        text_embeds = torch.cat(text_embed_list, dim=0)[task_ids]
        text_masks = torch.cat(text_mask_list, dim=0)[task_ids]
        return text_embeds, text_masks
    
    def prepare_batch(self, batch):
        new_batch = {
            "patch": batch["patch"].contiguous(), 
            "slot": batch["slot"].contiguous(),
            "state": batch["state"].contiguous(),
            "action": batch["action"].contiguous(),
            "valid_mask": batch["valid_mask"].contiguous().unsqueeze(-1),
            "text": self.text_embeds[batch["task_index"].contiguous().to(torch.long)],
            "att_mask": self.attention_mask[batch["task_index"].contiguous().to(torch.long)].flatten(0, 1) ## B, L, N, N -> BL, N, N
        }
        return new_batch
    
    def forward(self, batch):
        batch_action_chunk, att_map = self.simple_grp(batch)
        return batch_action_chunk, att_map

if __name__ == "__main__":
    from utils import save_checkpoint
    from hf_utils import create_model_repo, push_all_checkpoints

    config = tyro.cli(GRPConfig)
    set_seed_everywhere(config.seed)

    dataset_metadata = LeRobotDatasetMetadata(config.dataset_repo_id)
    config.tasks = dataset_metadata.tasks

    train_dataset = TensorDict.load_memmap(config.dataset_path)
    train_dataset = train_dataset.to(dtype=config.dtype).to(config.device)
    torch.cuda.empty_cache()

    save_log_freq = (train_dataset.batch_size[0] // config.batch_size)
    training_steps = save_log_freq * config.num_epochs

    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
        pin_memory=False,
        drop_last=True,
    )
    
    policy = CtrloGRPPolicy(config)
    policy = policy.to(config.dtype).to(config.device)
    torch.cuda.empty_cache()


    optimizer = optim.AdamW(policy.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=training_steps)
    loss_function = torch.nn.SmoothL1Loss(reduction="none")

    if config.torch_compile:
        policy = torch.compile(policy)

    # start a new wandb run to track this script
    if config.use_wandb:
        wandb.init(
            project = config.wproject,
            # track hyperparameters and run metadata
            config = config,
        )
        wandb.run.log_code(".")

    if config.push_to_hf:
        create_model_repo(config.hf_repo_id)
    policy.train()
    step = 0
    done = False
    while not done:
        for batch in train_dl:

            # train step
            optimizer.zero_grad()
            batch = policy.prepare_batch(batch)
            pred_action, _ =  policy(batch)
            loss = loss_function(pred_action, batch["action"])
            valid_mask = batch["valid_mask"]
            loss = loss * valid_mask
            loss = loss.sum() / valid_mask.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % config.log_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if  config.use_wandb:
                    wandb.log({"train_loss": loss.detach(), "learning_rate": current_lr}, step=step)

            if step % save_log_freq == 0:
                save_checkpoint(policy.simple_grp, save_dir=config.policy_save_path, step=step)
            step += 1
            if step >= training_steps:
                done = True
                break

    save_checkpoint(policy.simple_grp, save_dir=config.policy_save_path, step="final")
    if config.push_to_hf:
        push_all_checkpoints(config.policy_save_path, config.hf_repo_id)    
    
