"""
Simple GRP
A simple implementation of Octo using PyTorch. 
Based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
"""
import os
import psutil
import random
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import transformers
# from transformers import AutoTokenizer, AutoModel
from transformers import SiglipVisionModel, SiglipTextModel, SiglipImageProcessor, SiglipTokenizer
import wandb
# from tensordict import TensorDict

from accelerate import Accelerator
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from typing import Tuple


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
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.mha = nn.MultiheadAttention(inner_dim, heads, 
                                         dropout=0, bias=False, batch_first=True)

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, att_mask = None, return_att=False):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = qkv
        out, att = self.mha(q, k, v, need_weights=return_att, attn_mask=att_mask, average_attn_weights=False)

        return self.to_out(out), att    

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
    def forward(self, x, att_mask = None, return_att=False):
        attention_maps = []
        for attn, ff in self.layers:
            o, w  = attn(x, att_mask, return_att=return_att) 
            if return_att:
                attention_maps.append(w)
            x = o + x
            x = ff(x) + x
        return self.norm(x), tuple(attention_maps)


class SimpleGRP(nn.Module):
    def __init__(self, *, state_dim, action_dim, dim, depth, heads, mlp_dim, 
                 img_dim = 0, num_img_tokens=0, 
                 dim_head = 64, num_text_tokens = 0, num_state_tokens=0, num_action_tokens = 0):
        super().__init__()



        pos_embedding = create_sinusoidal_pos_embedding(
            num_positions = num_text_tokens + num_img_tokens + num_state_tokens + num_action_tokens,
            dimension = dim,
        ) 

        self.register_buffer("num_action_tokens", torch.tensor(num_action_tokens, dtype=torch.int64))
        self.register_buffer("pos_embedding", pos_embedding)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.img_projection = nn.Linear(img_dim, dim)

        self.state_projection = nn.Linear(state_dim, dim)
        self.action_token = nn.Embedding(num_action_tokens, dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, action_dim),
        )

    def forward(self, batch, return_att=False):
        batch_size = batch["image"].shape[0]
        device = batch["image"].device

        img_feat_tokens = self.img_projection(batch["image"])
        state_tokens = self.state_projection(batch["state"]).unsqueeze(1)
        action_tokens = self.action_token(torch.arange(self.num_action_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1))
        tokens = torch.cat([batch["text_tokens"], img_feat_tokens, state_tokens, action_tokens], dim=1)
        tokens = tokens + self.pos_embedding

        # print(tokens.shape, batch["att_mask"].shape)

        out, att_map = self.transformer(tokens, att_mask = batch["att_mask"], return_att=return_att)


        return self.action_head(out[:, -self.num_action_tokens:, :]), att_map

    @torch.no_grad()
    def get_action(self, batch):
        # batch size is expected to be 1
        # chunk size is self.num_action_tokens
        # returns the first predicted action in the chunk
        action_chunk, _ = self(batch)
        return action_chunk[0].float().cpu().numpy()
    
def init_model(transformer_params = {"depth": 6, "heads": 8, "mlp_dim": 2048}, 
               image_params = {"image_shape": (256, 256, 3), "patch_size": 16}, 
               text_token_shape = (0, 0, 0), action_shape = (0, 0), state_shape = (0, 0)):
    
    model = SimpleGRP(
        state_dim = state_shape[1],
        action_dim = action_shape[1],
        dim = text_token_shape[2],
        depth = transformer_params["depth"],
        heads = transformer_params["heads"],
        mlp_dim = transformer_params["mlp_dim"],
        img_dim = image_params["img_dim"],
        num_img_tokens = image_params["num_img_tokens"],
        num_text_tokens=text_token_shape[1],
        num_state_tokens=state_shape[0],
        num_action_tokens=action_shape[0],
    )
    return model

def train_step(batch):
    model.train()
    optimizer.zero_grad()
    pred_action, _ =  model(batch)
    # print(pred_action.shape)
    loss = loss_function(pred_action, batch["action"])
    valid_mask = batch["valid_mask"]
    loss = loss * valid_mask
    loss = loss.sum() / valid_mask.sum()
    # loss.backward()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    return loss.detach()

@torch.no_grad()
def encode_img(b_img):
    transformed_img = img_transform(b_img).to(siglip_vision_model.dtype).to(siglip_vision_model.device)
    output = siglip_vision_model(pixel_values=transformed_img).last_hidden_state
    # print(output.shape)
    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
    return output

@torch.no_grad()
def encode_txt(s, text_tokenizer=None, text_model=None):
    tokenized = text_tokenizer(s, return_tensors="pt", padding="max_length", return_attention_mask=True)
    text_embeds = text_model(input_ids=tokenized.input_ids).last_hidden_state
    return text_embeds, tokenized.attention_mask


def get_task_text_embeds(tasks, text_tokenizer=None, text_model=None):
    text_embed_list = []
    text_mask_list = []
    task_ids = []

    ## did a loop because I am not sure if items returned in order
    for task_id, task_text in tasks.items():
        task_ids.append(task_id)
        text_embed, text_mask = encode_txt(task_text, text_tokenizer, text_model)
        text_embed_list.append(text_embed)
        text_mask_list.append(text_mask)

    task_ids = torch.tensor(task_ids, dtype=torch.long)

    ## preserve order using task ids
    text_embeds = torch.cat(text_embed_list, dim=0)[task_ids]
    text_masks = torch.cat(text_mask_list, dim=0)[task_ids]
    return text_embeds, text_masks


## from openvla github
def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def evaluate_model(cfg):

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()  
    for task_id in range(task_suite.n_tasks):

        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        # if not ("moka" in task_description.lower()):
        #     continue
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {cfg.task_suite_name}, the " + \
            f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": cfg.img_shape[0],
            "camera_widths": cfg.img_shape[1]
        }
        print(env_args)
        # Get default LIBERO initial states
        env_init_states = task_suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        task_text_embeds, task_text_masks = encode_txt(task_description, siglip_tokenizer, siglip_text_model)

        task_text_masks = torch.cat([task_text_masks, torch.ones((task_text_masks.shape[0], cfg.num_img_tokens + cfg.num_state_tokens + cfg.num_action_tokens), dtype=task_text_masks.dtype)], dim=1)    
        task_attention_mask = get_att_mask(block_mask_arr.unsqueeze(0).repeat(task_text_masks.shape[0], 1), task_text_masks)
        task_attention_mask = torch.logical_not(task_attention_mask) # for nn.MHA
        task_attention_mask = task_attention_mask.unsqueeze(1).repeat(1, cfg.transformer_heads, 1, 1)
        # print(num_tokens, task_attention_mask.shape)

        task_text_embeds, task_attention_mask = task_text_embeds.to(cfg.dtype).to(cfg.device), task_attention_mask.to(cfg.dtype).to(cfg.device)
        num_success = 0
        batch_videos_np = np.zeros((cfg.num_trials, 500, cfg.img_shape[2], cfg.img_shape[0], cfg.img_shape[1]), dtype=np.uint8)
        for t in range(cfg.num_trials):
            #rd_task_init_state = env_init_states[np.random.randint(0, len(env_init_states))]
            task_init_state = env_init_states[t % len(env_init_states)]
            frames, cum_reward = run_env(env, task_init_state, cfg.num_env_steps, cfg.num_ol_actions,
                                        task_text_embeds, task_attention_mask)
            # print("eval cum reward:", cum_reward, "step:", step, "batch*step:", step*batch_size)
            num_success += cum_reward
            frames_np = np.transpose(frames, axes=(0,3,1,2))
            batch_videos_np[t, :frames_np.shape[0]] = frames_np
        num_acc = num_success / cfg.num_trials
        if cfg.use_wandb:
            wandb.log({
                        "video": wandb.Video(batch_videos_np, 
                        caption=task_description, fps=10, format="mp4"),
                        "task success": num_acc, "task id": task_id})
    

        env.close() 

@torch.no_grad()
def run_env(env, task_init_state, num_env_steps, num_ol_actions,
             task_text_embeds, task_attention_mask):
    
    dtype, device = task_attention_mask.dtype, task_attention_mask.device
    model.eval()
    env.reset()
    ## taken from https://github.com/Physical-Intelligence/openpi/blob/36dc3c037eb8a3921be9ecb94369d60cbf56f58f/examples/libero/main.py
    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
    # and we need to wait for them to fall
    obs = env.set_init_state(task_init_state)
    for _ in range(10):
        obs, reward, done, info = env.step([0.0] * 6 + [-1.0]) ## dummy action

    done, step, frame_list, cum_reward = False, 0, [], 0
    action_queue = []
    while not done and step < num_env_steps:
        image = obs["agentview_image"][::-1, ::-1].copy()
        # print(image.shape)
        frame_list.append(image)
        if not action_queue:
            state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]], axis=0)
            image_feat = encode_img(image)
            batch = {"image" : image_feat.to(dtype).to(device),
                     "state" : torch.tensor(state).unsqueeze(0).to(device, dtype=dtype),
                     "text_tokens": task_text_embeds.clone(),
                     "att_mask": task_attention_mask.clone().reshape(-1, num_tokens, num_tokens)
                    }
            pred_action =  model.get_action(batch)
            action_queue.extend(list(pred_action[:num_ol_actions]))
        obs, reward, done, info = env.step(action_queue.pop(0)) 
        cum_reward += reward
        step += 1

    return np.stack(frame_list, axis=0), cum_reward

def print_memory_usage():
    print('Memory Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,1), 'MB')
    print('Memory Cached:   ', round(torch.cuda.memory_reserved(0)/1024**2,1), 'MB')
    mem_in_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"Memory usage: {mem_in_mb:.2f} MB")

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, repo_id, delta_timestamps, device, dtype):
        dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

        loader = DataLoader(dataset,
                            batch_size=4096,
                            shuffle=False,
                            # drop_last=True,
                            )
        images = []
        states = []
        valid_masks = [] 
        actions = []
        task_indices = []
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        for item in tqdm(loader):
            images.append(normalize(item["observation.images.image"]).to(dtype=dtype).to(device))
            states.append(item["observation.state"].to(dtype=dtype).to(device))
            valid_masks.append((~item["action_is_pad"]).to(dtype=dtype).to(device))
            actions.append(item["action"].to(dtype=dtype).to(device))
            task_indices.append(item["task_index"])
        
        self.images = torch.cat(images, dim=0)
        self.states = torch.cat(states, dim=0)
        self.valid_masks = torch.cat(valid_masks, dim=0)
        self.actions = torch.cat(actions, dim=0)
        self.task_indices = torch.cat(task_indices, dim=0).to(dtype=torch.long)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "state": self.states[idx],
            "action": self.actions[idx],
            "valid_mask": self.valid_masks[idx],
            "task_index": self.task_indices[idx],
        }

@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    wproject: str = "sgrp-libero10img"
    siglip_model_name: str = "google/siglip-base-patch16-224"
    repo_id: str = "lerobot/libero_10_image"
    task_suite_name: str = "libero_10"

    learning_rate: float = 3e-4
    batch_size: int = 512
    training_epochs: int = 10

    transformer_mlp_dim: int = 2048
    transformer_depth: int = 6
    transformer_heads: int = 8

    num_state_tokens: int = 1
    num_action_tokens: int = 32

    use_wandb: bool = True
    torch_compile: bool = False
    do_eval_on_env: bool = True

    num_trials: int = 50
    num_env_steps: int = 500
    num_ol_actions: int = 8

    grp_model_name: str = "siglip_p16b256_grp_final.pt"
    model_path_base: str = "/network/scratch/o/ozgur.aslan/simple_grp"

    num_text_tokens = 64


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    print(cfg)

    model_path = os.path.join(cfg.model_path_base, cfg.grp_model_name) # f"/network/scratch/o/ozgur.aslan/simple_grp/{grp_model_name}"

    print("Doing Eval:", cfg.do_eval_on_env, "seed:", cfg.seed)
    set_seed_everywhere(cfg.seed)

    siglip_img_processor = SiglipImageProcessor.from_pretrained(cfg.siglip_model_name, do_rescale=False)
    siglip_tokenizer = SiglipTokenizer.from_pretrained(cfg.siglip_model_name)
    siglip_text_model = SiglipTextModel.from_pretrained(cfg.siglip_model_name, attn_implementation="sdpa")
    siglip_vision_model = SiglipVisionModel.from_pretrained(cfg.siglip_model_name, attn_implementation="sdpa")
    siglip_text_model.eval()
    siglip_vision_model.eval()

    img_transform = lambda imgs: siglip_img_processor(images=imgs, return_tensors="pt").pixel_values # imgs 

    cfg.num_img_tokens = (siglip_vision_model.config.image_size // siglip_vision_model.config.num_patches) ** 2
    cfg.img_dim = siglip_vision_model.config.hidden_size
    num_tokens = cfg.num_text_tokens + cfg.num_img_tokens + cfg.num_state_tokens + cfg.num_action_tokens


    block_mask_arr = torch.tensor([1] + (cfg.num_text_tokens + cfg.num_img_tokens - 1) * [0] \
                                + [1] + (cfg.num_state_tokens - 1) * [0] \
                                + [1] + (cfg.num_action_tokens - 1) * [0])

    if not cfg.do_eval_on_env:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

        dataset_metadata = LeRobotDatasetMetadata(cfg.repo_id)
        ds_features = dataset_metadata.features
        print(f"Dataset metadata: {dataset_metadata}")
        print(f"Dataset features: {ds_features}")

        delta_timestamps = {
            "action": [t / dataset_metadata.fps for t in range(cfg.num_action_tokens)],
        }

        train_dataset = DummyDataset(cfg.repo_id, delta_timestamps=delta_timestamps, device=cfg.device, dtype=cfg.dtype)
        print_memory_usage()
        # train_dataset = train_dataset.to(dtype=dtype).to(device)
        # torch.cuda.empty_cache()
        # print_memory_usage()
        print(train_dataset)
        save_log_freq = (len(train_dataset) // cfg.batch_size) 
        training_steps = (len(train_dataset) // cfg.batch_size) * cfg.training_epochs
        print(save_log_freq, training_steps)
        train_dl = DataLoader(train_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            drop_last=True,
                            )
           
        text_embeds, text_masks = get_task_text_embeds(dataset_metadata.tasks, siglip_tokenizer, siglip_text_model)
        text_embeds = text_embeds.to(dtype=cfg.dtype).to(cfg.device)
        del siglip_text_model, siglip_tokenizer
        torch.cuda.empty_cache()
   
        ## precompute blockwise attention mask
        text_masks = torch.cat([text_masks, torch.ones((text_masks.shape[0], cfg.num_img_tokens + cfg.num_state_tokens + cfg.num_action_tokens), dtype=text_masks.dtype)], dim=1)

        attention_mask = get_att_mask(block_mask_arr.unsqueeze(0).repeat(text_masks.shape[0], 1), text_masks)
        attention_mask = torch.logical_not(attention_mask) # for nn.MHA
        attention_mask = attention_mask.unsqueeze(1).repeat(1, cfg.transformer_heads, 1, 1)
        num_tokens = attention_mask.shape[-1]

        attention_mask = attention_mask.to(dtype=cfg.dtype).to(cfg.device)

    model_params = dict(transformer_params = {"depth": cfg.transformer_depth, 
                                              "heads": cfg.transformer_heads, 
                                              "mlp_dim": cfg.transformer_mlp_dim}, 
                        image_params =  {"img_dim": cfg.img_dim, 
                                         "num_img_tokens": cfg.num_img_tokens,
                                        }, 
                        text_token_shape = (0, cfg.num_text_tokens, cfg.img_dim),
                        state_shape = (cfg.num_state_tokens, 8), 
                        action_shape = (cfg.num_action_tokens, 7))
    


    model = init_model(**model_params) # .to(dtype=dtype).to(device=device)
    print_memory_usage()
    if cfg.do_eval_on_env:
        model.load_state_dict(torch.load(model_path))
        optimizer = None
        loss_function = None
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=training_steps)
        loss_function = torch.nn.SmoothL1Loss(reduction="none")

    if cfg.torch_compile:
        model = torch.compile(model)

    config = {
              **model_params, 
              "action_head": "2layer-mlp",
              "training_steps": training_steps,
              "loss_function": type(loss_function).__name__, 
              "optimizer": type(optimizer).__name__,
            } 
    
    # start a new wandb run to track this script
    if cfg.use_wandb:
        wandb.init(
            project = cfg.wproject,
            # track hyperparameters and run metadata
            config = config,
        )
        wandb.run.log_code(".")


    # mem_in_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    # print(f"Memory usage: {mem_in_mb:.2f} MB")

    if cfg.do_eval_on_env:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero_utils import quat2axisangle
        evaluate_model(cfg)
    else:
        accelerator = Accelerator(mixed_precision="bf16")
        model, siglip_vision_model, optimizer, train_dl = accelerator.prepare(model, siglip_vision_model, optimizer, train_dl)
        step = 0
        done = False
        while not done:
            for batch in tqdm(train_dl):
                torch.cuda.empty_cache()
                # batch = batch.to(device)
                batch = {"image": encode_img(batch["image"]), 
                         "state": batch["state"],
                         "action": batch["action"],
                         "valid_mask": batch["valid_mask"].unsqueeze(-1),
                         "text_tokens": text_embeds[batch["task_index"]],
                         "att_mask": attention_mask[batch["task_index"]].reshape(-1, num_tokens, num_tokens)
                        }

                loss = train_step(batch)
                if step % cfg.log_freq == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    if cfg.use_wandb:
                        wandb.log({"train loss": loss, "learning rate": current_lr}, step=step*cfg.batch_size)

                if step % save_log_freq == 0:
                    torch.save(model.state_dict(), model_path + str(step) + ".pt")
                    print("Model saved to", model_path + str(step) + ".pt")
                step += 1
                if step >= training_steps:
                    done = True
                    break

        torch.save(model.state_dict(), model_path + "final.pt")
    
