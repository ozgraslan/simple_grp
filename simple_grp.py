"""
Simple GRP
A simple implementation of Octo-like policy with PyTorch. 
Based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
"""
import os
import math
import numpy as np
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from einops import rearrange
from einops.layers.torch import Rearrange

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor, AutoBackbone
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import wandb

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

## this function is taken from https://github.com/huggingface/lerobot/blob/145fe4cd17971ea27188df23b2fe9d87e8e5128b/lerobot/common/policies/act/modeling_act.py#L685
def create_sinusoidal_pos_embedding(num_positions, dimension):
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()

## adapted from https://github.com/Physical-Intelligence/openpi/blob/36dc3c037eb8a3921be9ecb94369d60cbf56f58f/src/openpi/models/pi0.py#L20
def get_att_mask(mask_arr, inp_mask):
    cumsum = torch.cumsum(mask_arr, dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = inp_mask[:, None, :] * inp_mask[:, :, None]
    ## need to add the diagonal to the mask
    ## because older torch versions give nans 
    ## if a token does not attend to anything
    return torch.logical_or(torch.logical_and(attn_mask, valid_mask),  
                            torch.eye(valid_mask.shape[1], device=mask_arr.device).unsqueeze(0))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, att_mask = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)


        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask = att_mask, dropout_p = 0, is_causal = False)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, att_mask = None):
        for attn, ff in self.layers:
            x = attn(x, att_mask) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleGRP(nn.Module):
    def __init__(self, *, image_size, patch_size, state_dim, action_dim, dim, depth, heads, mlp_dim, 
                 channels = 3, dim_head = 64, num_text_tokens = 0, num_state_tokens=0, num_action_tokens = 0):
        super().__init__()
        patch_height, patch_width = pair(patch_size)


        image_height, image_width = pair(image_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        num_image_tokens = (image_height // patch_height) * (image_width // patch_width)

        pos_embedding = create_sinusoidal_pos_embedding(
            num_positions = num_text_tokens + num_image_tokens + num_state_tokens + num_action_tokens,
            dimension = dim,
        ) 
        self.register_buffer("pos_embedding", pos_embedding)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.state_projection = nn.Linear(state_dim, dim)
        self.action_token = nn.Embedding(1, dim)
        self.action_head = nn.Linear(dim, action_dim)

    def forward(self, batch):
        batch_size = batch["image"].shape[0]
        device = batch["image"].device

        img_tokens = self.to_patch_embedding(batch["image"])
        state_tokens = self.state_projection(batch["state"]).unsqueeze(1)
        action_tokens = self.action_token(torch.zeros((batch_size, 1), dtype=torch.long, device=device))
        tokens = torch.cat([batch["text_tokens"], img_tokens, state_tokens, action_tokens], dim=1)
        tokens = tokens + self.pos_embedding

        out = self.transformer(tokens, att_mask = batch["att_mask"])

        return self.action_head(out[:,-1,:])
    
def init_model(transformer_params = {"depth": 6, "heads": 8, "mlp_dim": 2048}, 
               image_params = {"image_shape": (256, 256, 3), "patch_size": 16}, 
               text_token_shape = (0, 0, 0), action_shape = (0, 0), state_shape = (0, 0)):
    
    model = SimpleGRP(
        image_size = image_params["image_shape"][0],
        patch_size = image_params["patch_size"],
        state_dim = state_shape[1],
        action_dim = action_shape[1],
        dim = text_token_shape[2],
        depth = transformer_params["depth"],
        heads = transformer_params["heads"],
        mlp_dim = transformer_params["mlp_dim"],
        channels = image_params["image_shape"][2],
        num_text_tokens=text_token_shape[1],
        num_state_tokens=state_shape[0],
        num_action_tokens=action_shape[0],
    )
    return model

def train_step(batch):
    model.train()
    optimizer.zero_grad()
    pred_action =  model(batch)
    loss = loss_function(pred_action, batch["action"])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


@torch.no_grad()
def validate_model():
    model.eval()
    loss_cumsum = 0
    for batch in val_dl:
        batch = {"image" : batch["observation.images.image"].to(device), 
                "state" : batch["observation.state"].to(device), 
                "action": batch["action"].to(device),
                "text_tokens": text_embeds[batch["task_index"].to(device)],
                "att_mask": attention_mask[batch["task_index"].to(device)]}

        pred_action =  model(batch)
        loss = loss_function(pred_action, batch["action"]) 
        loss_cumsum = loss_cumsum + loss.item()
    return loss_cumsum / len(val_dl)

@torch.no_grad()
def encode_txt(s):
    tokenized = t5_tokenizer(s, return_tensors="pt", padding="max_length", max_length=32)
    text_embeds = t5_model.encoder(input_ids=tokenized.input_ids).last_hidden_state
    return text_embeds, tokenized.attention_mask

def get_task_text_embeds(tasks):
    t5_model.eval()
    text_embed_list = []
    text_mask_list = []
    task_ids = []

    ## did a loop because I am not sure if items returned in order
    for task_id, task_text in tasks.items():
        task_ids.append(task_id)
        text_embed, text_mask = encode_txt(task_text)
        text_embed_list.append(text_embed)
        text_mask_list.append(text_mask)

    task_ids = torch.tensor(task_ids, dtype=torch.long)

    ## preserve order using task ids
    text_embeds = torch.cat(text_embed_list, dim=0)[task_ids]
    text_masks = torch.cat(text_mask_list, dim=0)[task_ids]
    return text_embeds, text_masks

def get_eval_env(task_id = 0, image_shape=(256, 256, 3), seed=0):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": image_shape[0],
        "camera_widths": image_shape[1]
    }

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(task_id)
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description, initial_states


@torch.no_grad()
def evaluate_on_env(eval_task_init_state, eval_task_text_embeds, eval_task_attention_mask):
    model.eval()

    eval_env.reset()
    ## taken from https://github.com/Physical-Intelligence/openpi/blob/36dc3c037eb8a3921be9ecb94369d60cbf56f58f/examples/libero/main.py
    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
    # and we need to wait for them to fall
    obs = eval_env.set_init_state(eval_task_init_state)
    for _ in range(10):
        obs, reward, done, info = eval_env.step([0.0] * 6 + [-1.0])

    done, cum_reward, step, frame_list = False, 0, 0, []
    while not done and step < 500:
        image = rotate(obs["agentview_image"], angle=180)
        frame_list.append(image)
        state = np.concatenate([obs["robot0_eef_pos"], Rotation.from_quat(obs["robot0_eef_quat"], scalar_first=False).as_rotvec(), obs["robot0_gripper_qpos"]], axis=0)
        batch = {"image" : to_tensor(image).unsqueeze(0).to(device), 
                 "state" : torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device),
                 "text_tokens": eval_task_text_embeds.clone(),
                 "att_mask": eval_task_attention_mask.clone()
                }
        pred_action =  model(batch)
        obs, reward, done, info = eval_env.step(pred_action[0].cpu().numpy())
        cum_reward += reward
        step += 1

    return cum_reward, np.stack(frame_list, axis=0)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "lerobot/libero_10_image"
    patch_size = 16
    batch_size = 128
    training_steps = 100000
    log_freq = 1000
    val_log_freq = 2000
    eval_log_freq = 4000
    num_state_tokens = 1
    num_action_tokens = 1
    transformer_mlp_dim = 2048
    transformer_depth = 6
    transformer_heads = 8
    use_wandb = False
    compile = False
    learning_rate = 5e-5
    model_path = '/network/projects/real-g-grp/simple_grp_full.pt'

    # - Calculate train and val episodes
    dataset_metadata = LeRobotDatasetMetadata(repo_id)
    ds_features = dataset_metadata.features
    print(f"Dataset metadata: {dataset_metadata}")
    print(f"Dataset features: {ds_features}")

    total_episodes = dataset_metadata.total_episodes
    episodes = list(range(dataset_metadata.total_episodes))
    num_train_episodes = math.floor(total_episodes * 95 / 100)
    train_episodes = episodes[:num_train_episodes]
    val_episodes = episodes[num_train_episodes:]
    print(f"Number of episodes in full dataset: {total_episodes}")
    print(f"Number of episodes in training dataset (90% subset): {len(train_episodes)}")
    print(f"Number of episodes in validation dataset (10% subset): {len(val_episodes)}")
    # - Load train an val datasets
    train_dataset = LeRobotDataset(repo_id, episodes=train_episodes)
    val_dataset = LeRobotDataset(repo_id, episodes=val_episodes)
    print(f"Number of frames in training dataset (90% subset): {len(train_dataset)}")
    print(f"Number of frames in validation dataset (10% subset): {len(val_dataset)}")


    train_dl = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=device!="cpu",
                          drop_last=True,
                         )
    
    val_dl = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=device!="cpu",
                        drop_last=True,
                       ) # makes avg comp easier

    t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    t5_model = AutoModel.from_pretrained("google-t5/t5-small")
       
    text_embeds, text_masks = get_task_text_embeds(dataset_metadata.tasks)
    num_text_tokens = text_embeds.shape[1]
    text_embeds = text_embeds.to(device)
   
    ## precompute blockwise attention mask
    img_shape = ds_features["observation.images.image"]["shape"]
    num_img_tokens = (img_shape[0] // patch_size) * (img_shape[1] // patch_size)
    text_masks = torch.cat([text_masks, torch.ones((text_masks.shape[0], num_img_tokens + num_state_tokens + num_action_tokens), dtype=text_masks.dtype)], dim=1)
    block_mask_arr = torch.tensor([1] + (num_text_tokens + num_img_tokens  - 1) * [0] \
                                + [1] + (num_state_tokens - 1) * [0] \
                                + [1] + (num_action_tokens - 1) * [0])

    attention_mask = get_att_mask(block_mask_arr.unsqueeze(0).expand(text_masks.shape[0], -1), text_masks).unsqueeze(1)

    attention_mask = attention_mask.to(device)

    model_params = dict(transformer_params = {"depth": transformer_depth, 
                                              "heads": transformer_heads, 
                                              "mlp_dim": transformer_mlp_dim}, 
                        image_params = {"image_shape": img_shape,
                                        "patch_size": patch_size}, 
                        text_token_shape = tuple(text_embeds.shape),
                        state_shape = (num_state_tokens, ds_features["observation.state"]["shape"][0]), 
                        action_shape = (num_action_tokens, ds_features["action"]["shape"][0]))
    


    model = init_model(**model_params).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=training_steps)

    loss_function = torch.nn.SmoothL1Loss()

    if compile:
        model = torch.compile(model)

    eval_env, eval_task_text, eval_env_init_states = get_eval_env(task_id=0, image_shape=ds_features["observation.images.image"]["shape"], seed=0)
    eval_task_text_embeds, eval_task_text_masks = encode_txt(eval_task_text)
    eval_task_text_masks = torch.cat([eval_task_text_masks, torch.ones((eval_task_text_masks.shape[0], num_img_tokens + num_state_tokens + num_action_tokens), dtype=eval_task_text_masks.dtype)], dim=1)
    
    eval_task_attention_mask = get_att_mask(block_mask_arr.unsqueeze(0).expand(eval_task_text_masks.shape[0], -1), eval_task_text_masks).unsqueeze(1)

    eval_task_text_embeds, eval_task_attention_mask = eval_task_text_embeds.to(device), eval_task_attention_mask.to(device)

    config = {**model_params, "batch_size": batch_size, "training_steps": training_steps,  "lr": learning_rate,
              "loss_function": type(loss_function).__name__, "optimizer": type(optimizer).__name__,
              "log_freq": log_freq, "val_log_freq": val_log_freq}
    
    # start a new wandb run to track this script
    if use_wandb:
        wandb.init(
            project = "sgrp-libero10img",
            # track hyperparameters and run metadata
            config = config
        )
        wandb.run.log_code(".")
    step = 0
    done = False
    while not done:
        for batch in train_dl:
            batch = {"image" : batch["observation.images.image"].to(device), 
                        "state": batch["observation.state"].to(device),
                        "action": batch["action"].to(device),
                        "text_tokens": text_embeds[batch["task_index"].to(device)],
                        "att_mask": attention_mask[batch["task_index"].to(device)]}

            loss = train_step(batch)
            if step % log_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if use_wandb:
                    wandb.log({"train loss": loss, "learning rate": current_lr}, step=step*batch_size)
                print("train loss:", loss.item(), "current learning rate:", current_lr, 
                      "step:", step, "batch*step:", step*batch_size)

            if step % val_log_freq == 0:
                avg_val_loss = validate_model()
                if use_wandb:
                    wandb.log({"avg val loss": avg_val_loss}, step=step*batch_size)
                print("avg val loss:", avg_val_loss, "step:", step, "batch*step:", step*batch_size)
            
            if step % eval_log_freq == 0:
                rd_eval_task_init_state = eval_env_init_states[np.random.randint(0, len(eval_env_init_states))]
                cum_reward, frames = evaluate_on_env(rd_eval_task_init_state, eval_task_text_embeds, eval_task_attention_mask)
                print("eval cum reward:", cum_reward, "step:", step, "batch*step:", step*batch_size)
                if use_wandb:
                    wandb.log({"eval env return": cum_reward}, step=step*batch_size)
                    wandb.log({"video": wandb.Video(np.transpose(frames, axes=(0,3,1,2)), fps=dataset_metadata.fps)})
                

            step += 1
            if step >= training_steps:
                done = True
                break
    torch.save(model.state_dict(), model_path)
