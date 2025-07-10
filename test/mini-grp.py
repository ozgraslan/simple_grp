import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from einops import rearrange

# import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm, trange
import cv2
from scipy.spatial.transform import Rotation
import mani_skill

# data loading
def get_batch_grp(split, dataset, batch_size, history_length=0, n_chunk=1, device="cpu"):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    dataset_size = int(len(data["img"]))
    ix = np.random.randint(dataset_size, size=(batch_size,))
    x_list = []
    for h in range(history_length, -1, -1):
        ik = (ix - h)
        ik[ik < 0] = 0
        x_list.append(torch.tensor(data["img"][ik], dtype=torch.float))
    x = torch.stack(x_list, dim=1)
    x_goal = torch.tensor(data["goal"][ix])
    x_goal_img = torch.tensor(data["goal_img"][ix][:,None], dtype=torch.float).to(device)
    x_goal_mask = torch.tensor(data["goal_mask"][ix]).to(device)  if data["goal_mask"] is not None else None
    y_list = []
    for c in range(n_chunk):
        ik = ix+c
        ik[ik >= dataset_size] = dataset_size - 1
        y_list.append(torch.tensor(data["action"][ik]))
    y = torch.concatenate(y_list, dim=-1)
    return x.to(device), x_goal.to(device), x_goal_mask, x_goal_img, y.to(device)


@torch.no_grad()
def estimate_loss(model, device="cpu"):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_goal, x_goal_mask, x_goal_img, Y = get_batch_grp(split, model._dataset, model._cfg.batch_size, model._cfg.history_length, model._cfg.n_chunk, device)
            logits, loss = model(X, x_goal, x_goal_mask, x_goal_img, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_patches_fast(images, n_patches):
    height = images.shape[-2]
    patch_size = height // n_patches ## n_patches = 8

    patches = rearrange(images, 'b k (h p1) (w p2) c -> b (k h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    return patches

def get_patches_fast2(image_tokens):
    patches = rearrange(image_tokens, 'b k p c -> b (k p) c')
    return patches

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

## This is an encoder head (full attention)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        ### Block masked attention
        wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        b = x.shape[0]
        return x.reshape(b, *self.shape)

class GRP(nn.Module):
  def __init__(self, dataset, cfg, mlp_ratio=4):
    super(GRP, self).__init__()
    self._dataset = dataset
    self._cfg = cfg

    # Positional embedding
    self.register_buffer('positional_embeddings', calc_positional_embeddings((2+cfg.history_length)*cfg.n_patches ** 2 + 1 + cfg.block_size, cfg.n_embd), persistent=False)

    print(self.positional_embeddings.shape)
    # Learnable action token for action prediction
    self.action_token = nn.Embedding(1, cfg.n_embd)

    # Linear layer for patch processing
    if cfg.use_siglip:
        self.lin_patch = nn.Sequential(nn.Linear(cfg.vocab_size, cfg.n_embd), nn.ReLU(), nn.Linear(cfg.n_embd, cfg.n_embd))
    else:
        self.lin_patch = nn.Linear(cfg.image_shape[-1] * (cfg.image_shape[-2]//cfg.n_patches)**2, cfg.n_embd)


    # Learnable text tokens 
    if cfg.use_t5 or cfg.use_siglip:
        self.text_model = nn.Sequential(nn.Linear(cfg.vocab_size, cfg.n_embd), nn.ReLU(), nn.Linear(cfg.n_embd, cfg.n_embd))
    else:
        self.text_model = nn.Embedding(cfg.vocab_size, cfg.n_embd)

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, dropout=cfg.dropout) for _ in range(cfg.n_blocks)])

    # 5) Classification MLPk
    self.action_dim = 3 if cfg.only_pos else (cfg.action_dim+1)
    action_bins = 1 if not cfg.discretize_actions else cfg.action_bins       
    if cfg.use_mlp_head:
        action_head = [nn.Linear(cfg.n_embd, cfg.n_embd), 
                        nn.ReLU(), 
                        nn.Linear(cfg.n_embd, self.action_dim * action_bins * cfg.n_chunk)]
    else:
        action_head = [nn.Linear(cfg.n_embd, self.action_dim * action_bins * cfg.n_chunk)]
     
    if cfg.discretize_actions:
        self.mlp = nn.Sequential(*action_head, Reshape(shape=(action_bins, self.action_dim*cfg.n_chunk)))
        self.loss_func = nn.CrossEntropyLoss()
    else:
        self.mlp = nn.Sequential(*action_head)
        self.loss_func = nn.MSELoss()

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, images, goals_txt, goal_mask=None, goal_imgs=None, targets=None):
    # Dividing images into patches
    B, k = images.shape[:2]
    # B, T = goals_txt.shape
    ## Provide the logic to produce the output and loss for the GRP

    # Map the vector corresponding to each patch to the hidden size dimension
    # print(images.shape, goal_imgs.shape)
    if self._cfg.use_siglip:
        # combine history with tokens
        img_patches = get_patches_fast2(images)
        goal_img_patches = get_patches_fast2(goal_imgs)

    else:
        img_patches = get_patches_fast(images, self._cfg.n_patches)
        goal_img_patches = get_patches_fast(goal_imgs, self._cfg.n_patches)
    
    image_embeds = self.lin_patch(img_patches)
    # Adding classification and goal_img tokens to the tokens
    goal_img_embeds = self.lin_patch(goal_img_patches)

    # print(image_embeds.shape, goal_img_embeds.shape)

    goal_text_embeds = self.text_model(goals_txt)
    action_embeds = self.action_token(torch.zeros(B, 1, dtype=torch.long, device=images.device))

    n_gt_tokens, n_gi_tokens, n_i_tokens = goal_text_embeds.shape[-2], goal_img_embeds.shape[-2], image_embeds.shape[-2]
    embeds = torch.concatenate([goal_text_embeds, goal_img_embeds, image_embeds, action_embeds], dim=-2) 

    n_tokens = embeds.shape[-2]
    # Adding positional embedding
    out = embeds + self.positional_embeddings[:n_tokens].repeat(B, 1, 1)

    # print(n_gt_tokens, n_tokens)
    # Compute blocked masks
    blocked_masks = torch.eye(n_tokens, dtype=int, device=out.device)[None].repeat(B, 1, 1)

    if targets is not None and np.random.rand() < 0.0:
        ## use image_goal
        blocked_masks[:, n_gt_tokens:, n_gt_tokens:n_gt_tokens+n_gi_tokens] = 1
        # blocked_masks[:, n_gt_tokens+n_gi_tokens:, n_gt_tokens+n_gi_tokens:n_gt_tokens+n_gi_tokens+n_i_tokens] = 1
        # blocked_masks[:, n_gt_tokens+n_gi_tokens+n_i_tokens:, n_gt_tokens+n_gi_tokens+n_i_tokens:] = 1
    else:
        ## use text goal
        blocked_masks[:, :, :n_gt_tokens] = 1
        if self._cfg.use_t5:
            # text_mask = (~(goals_txt == 0)).int() ## 0 is the padding embed id
            ## following is done to mask out the padded text embeddings
            blocked_masks[:, : , :n_gt_tokens] = blocked_masks[:,:,:n_gt_tokens] & goal_mask[:,None]
            blocked_masks[:, :n_gt_tokens,:] = blocked_masks[:,:n_gt_tokens,:] & goal_mask[:,:,None]
            ## this one to make sure masked ones only attends to themselves, so we dont get nans
            blocked_masks[:, :n_gt_tokens, :n_gt_tokens] = blocked_masks[:,:n_gt_tokens,:n_gt_tokens] | torch.eye(n_gt_tokens, dtype=int, device=out.device)[None]

        blocked_masks[:, n_gt_tokens:n_gt_tokens+n_gi_tokens, :n_gt_tokens] = 0
    for ik in range(k):
        blocked_masks[:, n_gt_tokens+n_gi_tokens + ik*(n_i_tokens//k):, n_gt_tokens+n_gi_tokens+ ik*(n_i_tokens//k):n_gt_tokens+n_gi_tokens+ (ik+1)*(n_i_tokens//k)] = 1
    blocked_masks[:, n_gt_tokens+n_gi_tokens+n_i_tokens:, n_gt_tokens+n_gi_tokens+n_i_tokens:] = 1

    # Transformer Blocks
    for block in self.blocks:
        out = block(out, blocked_masks)

    # Getting the classification token only
    out = self.mlp(out[:,-1,:])
    # Compute output and loss
    loss = self.loss_func(out, targets) if targets is not None else None
    # print(out[..., :self.action_dim].shape, out.shape)
    return (out, loss)

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
    reserved = torch.cuda.memory_reserved() / (1024**2)  # Convert to MB
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Reserved Memory: {reserved:.2f} MB")

import hydra, json
from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="conf", config_name="grp-mini")
@hydra.main(config_path="conf", config_name="bridge-64-light")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    ## DONE: changed how to extract outputdir
    log_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    print ("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    cfg.device = device
    from datasets import load_dataset, concatenate_datasets
    dataset = concatenate_datasets([load_dataset(cfg.dataset.to_name, split=st_name, keep_in_memory=True) for st_name in ["train_1"]])
    dataset = dataset.select_columns(["img", "seg", "goal_img", "rel_tcp_pose", "action"])
    # dataset = dataset.rename_column("rel_tcp_pose", "action")
    new_column_values = ["put carrot on plate"] * len(dataset)  # Example: A constant value for all rows
    dataset = dataset.add_column("goal", new_column_values)
    print('Features:', dataset.features)
    # get first 100 samples
    # dataset = dataset.select(range(100))
    # print(len(dataset["img"][0]))

    # rel_tcp_pose = np.array(dataset["rel_tcp_pose"])
    # rel_tcp_pos = rel_tcp_pose[:, :3]
    # rel_tcp_quat = rel_tcp_pose[:, 3:] #wxyz
    # axis_angle = Rotation.from_quat(rel_tcp_quat, scalar_first=True).as_rotvec()
    # print(rel_tcp_quat.shape, axis_angle.shape)

    # open_gripper = np.array(dataset["action"])[:, -1:]
# np.concatenate((rel_tcp_pos
#                                 ,axis_angle
#                                 ,open_gripper
#                                 ), axis=1)
    print(np.array(dataset["action"]).shape)
    if cfg.use_segmentation:
        img = np.expand_dims(np.array(dataset["seg"]), axis=-1)
        img = (img * np.array([11, 61, 127])).astype(np.uint8)
        print(img.shape)
    else:
        img = np.array(dataset["img"])
    dataset_tmp = {
        "img": img,
        "action": np.array(dataset["action"]),
        "goal_img": np.array(dataset["goal_img"]),
        "goal": dataset["goal"]
    }
    shortest_text_len = min([len(txt) for txt in dataset_tmp["goal"]])
    longest_text_len = max([len(txt) for txt in dataset_tmp["goal"]])
    del dataset

    print("shortest text len:", shortest_text_len)
    print("longest text len:", longest_text_len)

    if cfg.use_t5:
        from transformers import T5Tokenizer, T5Model
        t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        t5_model = T5Model.from_pretrained("google-t5/t5-small")
        for param in t5_model.parameters():
            param.requires_grad = False
        text_len = longest_text_len

        def encode_txt(s):
            tokenized = t5_tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=cfg.block_size)
            text_embeds = t5_model.encoder(input_ids=tokenized.input_ids).last_hidden_state
            return text_embeds.numpy(), tokenized.attention_mask.numpy()
        
        encoded_text, att_mask = encode_txt(dataset_tmp["goal"])
        # print(encoded_text.shape, att_mask.shape)
        cfg.vocab_size = t5_model.config.d_model
    elif cfg.use_siglip:
        from transformers import Siglip2TextModel, AutoTokenizer
        ckpt = "google/siglip2-so400m-patch16-256"
        sl_tokenizer = AutoTokenizer.from_pretrained(ckpt)
        slt_model = Siglip2TextModel.from_pretrained(ckpt, device_map="auto").eval()
        for param in slt_model.parameters():
            param.requires_grad = False

        @torch.no_grad()
        def encode_txt(s):
            if type(s) is list and len(s) > 1 :
                print("Number of text goals:", len(s))
                bs = 4096
                lst = []
                lst2 = []
                print("iter num", len(s)//bs)
                for i in range(len(s)//bs):

                    # print(i)
                    # print_memory_usage()
                    batch_inputs = sl_tokenizer(s[i*bs:(i+1)*bs], return_tensors="pt", padding="max_length", max_length=64).to(slt_model.device)
                    batch_embeddings = slt_model(**batch_inputs).last_hidden_state.cpu().numpy().astype(np.float32)  
                    batch_masks = (~(batch_inputs.input_ids==0)).to(torch.int64).cpu().numpy()

                    lst.append(batch_embeddings)
                    lst2.append(batch_masks)
                    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter

                batch_inputs = sl_tokenizer(s[(len(s)//bs)*bs:], return_tensors="pt", padding="max_length", max_length=64).to(slt_model.device)
                batch_embeddings = slt_model(**batch_inputs).last_hidden_state.cpu().numpy().astype(np.float32) 
                batch_masks = (~(batch_inputs.input_ids==0)).to(torch.int64).cpu().numpy()
                
                lst.append(batch_embeddings)
                lst2.append(batch_masks)
                torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter

                text_embeddings = np.concatenate(lst, axis=0)
                txt_masks = np.concatenate(lst2, axis=0)
            else:
                txt_inputs = sl_tokenizer(s, return_tensors="pt", padding="max_length", max_length=64).to(slt_model.device)
                text_embeddings = slt_model(**txt_inputs).last_hidden_state.cpu().numpy() 
                txt_masks = (~(txt_inputs.input_ids==0)).to(torch.int64).cpu().numpy()
                torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter

            return text_embeddings, txt_masks

        dataset_tmp["goal"] = [txt.lower() for txt in dataset_tmp["goal"]]
        encoded_text, att_mask = encode_txt(dataset_tmp["goal"])
        # print(encoded_text.dtype, att_mask.dtype)
        cfg.vocab_size = encoded_text.shape[-1]
        cfg.block_size = 64
    else:
        cfg.block_size = shortest_text_len
        text_len = shortest_text_len
        # here are all the unique characters that occur in this text
        chars = sorted(list(set([item for row in dataset_tmp["goal"] for item in row]))) ## Flatten to a long string
        cfg.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode_txt = lambda s: [stoi[c] for c in s] # text encoder to tokens: 
        decode_txy = lambda l: ''.join([itos[i] for i in l]) # token decoder to text: 
        print("vocab_size:", cfg.vocab_size)
        print("example text encode:", dataset_tmp["goal"][0], encode_txt(dataset_tmp["goal"][0]))
        encoded_text = [encode_txt(goal[:text_len]) for goal in dataset_tmp["goal"]]

    action_mean, action_std = dataset_tmp["action"].mean(axis=0, keepdims=True), dataset_tmp["action"].std(axis=0, keepdims=True)
    action_low, action_high = np.percentile(dataset_tmp["action"], 1, axis=0, keepdims=True), np.percentile(dataset_tmp["action"], 99, axis=0, keepdims=True)
    act_dim = 3 if cfg.only_pos else cfg.action_dim + 1

    if cfg.discretize_actions:
        ## to discretize, quantile method from RT2 and OpenVLA is used.
        ## that is the bins are computed from the data between the 1st and 99th quantile
        ## this will remove the outlier actions

        ## using bins + 1 so actions will index mean values of bins
        space = np.linspace(-1-1e-6, 1+1e-6, cfg.action_bins+1)
        mean_space = (space[1:] + space[:-1]) / 2.0
        ## first clip actions to be in (low, high)
        ## then digitize actions. 
        ## normally if values are lower than space[0] they will be assigned 0 
        ## and if values are higher then space[-1] they will be assigned to len(space) = n_bins+1
        ## but with clipping they will be in [1, n_bins] 
        ## so we need to extract 1 from digitized actions to make them in [0, n_bins-1]
        encode_action = lambda af: (np.digitize((2 * ((np.clip(af, action_low, action_high) - action_low) / (action_high-action_low)) - 1), space) - 1).astype(np.int32)
        decode_action = lambda af: (((action_high-action_low) * 0.5* (mean_space[af.argmax(axis=1)] +1)) + action_low)
    else:
        ## normalize actions to be mean = 0 and std = 1
        ## try quantile normalization [-1, 1]
        if cfg.normalize_normal:
            encode_action = lambda af: ((af - action_mean) / (action_std + 1e-8)).astype(np.float32) #[:,:act_dim]
            decode_action = lambda af: (af * action_std + action_mean).astype(np.float32) #[:,:act_dim]
        else:
            encode_action = lambda af: (2 * ((np.clip(af, action_low, action_high) - action_low) / (action_high-action_low)) - 1).astype(np.float32)
            decode_action = lambda af: ((action_high-action_low) * ((af + 1) / 2.0) + action_low).astype(np.float32)

    ## Get images and encode them to map to [-1, 1]
    if cfg.use_siglip:
        cfg.n_patches = 16 # spesific to the model
        from transformers import SiglipVisionModel, AutoProcessor
        ckpt = "google/siglip2-so400m-patch16-256"
        sl_processor = AutoProcessor.from_pretrained(ckpt)
        slv_model = SiglipVisionModel.from_pretrained(ckpt, device_map="auto").eval()

        for param in slv_model.parameters():
            param.requires_grad = False

        @torch.no_grad()
        def encode_state(sf):
            if sf.shape[0] > 1 :
                bs = 1024
                lst = []
                print("img device", slv_model.device, "img iter", sf.shape[0]//bs)
                for i in range(sf.shape[0]//bs):
                    # print(i)
                    print_memory_usage()

                    img_inputs = sl_processor(images=sf[i*bs:(i+1)*bs], return_tensors="pt").to(slv_model.device)
                    batch_embeddings = slv_model(**img_inputs).last_hidden_state.cpu().numpy().astype(np.float32)  
                    lst.append(batch_embeddings)
                    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
                
                img_inputs = sl_processor(images=sf[(sf.shape[0]//bs)*bs:], return_tensors="pt").to(slv_model.device)
                batch_embeddings = slv_model(**img_inputs).last_hidden_state.cpu().numpy().astype(np.float32)  

                lst.append(batch_embeddings)
                torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
                image_embeddings = np.concatenate(lst, axis=0)
            else:
                img_inputs = sl_processor(images=sf, return_tensors="pt").to(slv_model.device)
                image_embeddings = slv_model(**img_inputs).last_hidden_state.cpu().numpy().astype(np.float32) 
                torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter

            return image_embeddings

        resize_state = lambda sf: sf
    else:
        encode_state = lambda sf:   ((sf/(255.0)*2.0)-1.0).astype(np.float32) # encoder: take a float, output an integer
        def resize_state(sf):
            arr = []
            for s in sf:
                arr.append(cv2.resize(s.astype(np.uint8), (cfg.image_shape[0], cfg.image_shape[1])))
            return np.array(arr) # resize state
        # resize_state = lambda sf:   np.array([cv2.resize(s, (cfg.image_shape[0], cfg.image_shape[1])) for s in sf])  # resize state

    n = int(0.95*len(dataset_tmp["img"])) # first 95% will be train, rest val
    dataset_tmp = { 
        "train":
            {
            "img": torch.tensor(encode_state(dataset_tmp["img"][:n])),
            "action": torch.tensor(encode_action(dataset_tmp["action"][:n]), dtype=torch.float if not cfg.discretize_actions else torch.long),            
            "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"][:n])),
            "goal": torch.tensor(encoded_text[:n], dtype=torch.float if cfg.use_t5 or cfg.use_siglip else torch.long),
            "goal_mask": torch.tensor(att_mask[:n], dtype=torch.int) if cfg.use_t5 or cfg.use_siglip else None
            },
        "test": 
        {
            "img": torch.tensor(encode_state(dataset_tmp["img"][n:])),
            "action": torch.tensor(encode_action(dataset_tmp["action"][n:]), dtype=torch.float if not cfg.discretize_actions else torch.long),            
            "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"][n:])),
            "goal": torch.tensor(encoded_text[n:], dtype=torch.float if cfg.use_t5 or cfg.use_siglip else torch.long),
            "goal_mask": torch.tensor(att_mask[n:], dtype=torch.int) if cfg.use_t5 or cfg.use_siglip else None

        }
    }

    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg)
        )
        wandb.run.log_code(".")
    model = GRP(dataset_tmp, cfg)
    # model = torch.compile(model)
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)

    if cfg.simEval:
        # import simpler_env
        from gymnasium.wrappers import FrameStackObservation
        import gymnasium as gym
        # task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env

        env = gym.make(
            "PutCarrotOnPlateInSceneSep-v1",
            obs_mode="rgb+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="dense",
            sensor_configs=dict(shader_pack="default"),
            human_render_camera_configs=dict(shader_pack="default"),
            viewer_camera_configs=dict(shader_pack="default"),
            sim_backend="cpu",
        )
        env = FrameStackObservation(env, cfg.history_length+1) # +1 for the current step observation
        env_unwrapped = env.unwrapped ## Updated gymnasium wrapper adds lots of wrappers.

    for iter in range(cfg.max_iters):

        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, device=device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if not cfg.testing:
                wandb.log({"train loss": losses['train'], "val loss": losses['val']})

            if cfg.simEval and (iter % cfg.eval_vid_iters == 0): ## Do this eval infrequently because it takes a fair bit of compute
                rewards = []
                for j in range(cfg.sim.eval_episodes): ## Better to eval over a few different goal configurations
                    obs, reset_info = env.reset()
                    instruction = env_unwrapped.get_language_instruction()
                    # instruction = [instruction] if cfg.use_t5 or cfg.use_siglip else instruction
                    print("Reset info", reset_info)
                    print("Instruction", instruction)
                    frames = []
                    done, truncated, timeLimit, t = False, False, 100, 0
                    while not (done or truncated or (t > timeLimit)):
                        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
                        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
                        if cfg.use_segmentation:
                            seg = obs["sensor_data"]["3rd_view_camera"]["segmentation"][:,0] 
                            image = (seg * np.array([11, 61, 127])).astype(np.uint8)
                        else:
                            image = obs["sensor_data"]["3rd_view_camera"]["rgb"][:,0]
                        # image = image[:,:,:3] ## Remove last dimension of image color
                        # with torch.no_grad():
                        if cfg.use_t5 or cfg.use_siglip:
                            encoded_text, text_mask = encode_txt(instruction)  
                        else: 
                            encoded_text =np.array([encode_txt(instruction[0][:text_len])])
                            text_mask = None
                        # print(encoded_text)
                        # print(encoded_text.shape)
                        action, loss = model.forward(torch.tensor(encode_state(resize_state(image))[None]).to(device)
                                                ,torch.tensor(encoded_text).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                                                ,torch.tensor(text_mask).to(device) if text_mask is not None else None
                                                ,torch.tensor(encode_state(resize_state(image))[:1][None]).to(device) ## Not the correct goal image... Should mask this.
                                                )
                        # print(action.shape)
                        if cfg.n_chunk > 1 and cfg.use_more_actions:
                            for chunk in range(cfg.n_chunk-1):
                                action_chunk = decode_action(action[:, chunk*(cfg.action_dim+1):(chunk+1)*(cfg.action_dim+1)].cpu().detach().numpy())[0]
                                obs, reward, done, truncated, info = env.step(action_chunk)
                                if cfg.use_segmentation:
                                    seg = obs["sensor_data"]["3rd_view_camera"]["segmentation"][:,0] 
                                    image = (seg * np.array([11, 61, 127])).astype(np.uint8)
                                else:
                                    image = obs["sensor_data"]["3rd_view_camera"]["rgb"][:,0]
                                # reward = -np.linalg.norm(info["eof_to_obj1_diff"])
                                frames.append(image[-1])
                                rewards.append(reward)
                            action = action[:, -1*(cfg.action_dim+1):]

                        if cfg.only_pos:
                            action = np.concatenate((decode_action(action.cpu().detach().numpy()), [[0, 0, 0, 0]]), axis = -1)[0]
                        else:
                            action = decode_action(action.cpu().detach().numpy())[0] ## Add in the gripper close action
                            # print(action.shape)

                        obs, reward, done, truncated, info = env.step(action)
                        # reward = -np.linalg.norm(info["eof_to_obj1_diff"])
                        frames.append(image[-1])
                        rewards.append(reward)
                        t=t+1
                
                episode_stats = info.get('episode_stats', {})
                print("Episode stats", episode_stats)
                print(f"avg reward {np.mean(rewards):.8f}")
                if not cfg.testing:
                    wandb.log({"avg reward": np.mean(rewards)})
                import moviepy as mpy
                clip = mpy.ImageSequenceClip(list(frames), fps=20)
                clip.write_videofile("./sim-env-"+str(iter)+".mp4", fps=20)
                if not cfg.testing:
                    wandb.log({"example": wandb.Video("./sim-env-"+str(iter)+".mp4")})

        # sample a batch of data
        xb, xg, xgm, xgi, yb = get_batch_grp('train', dataset_tmp, cfg.batch_size, cfg.history_length, cfg.n_chunk, device=device)

        # evaluate the loss
        _, loss = model(xb, xg, xgm, xgi, yb)
        loss.backward()

        if (iter + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            # scheduler.step() ## Added scheduler
            optimizer.zero_grad(set_to_none=True)

    if not cfg.testing:
        wandb.finish()
    return losses['val']

if __name__ == "__main__":
    results = my_main()
    print("results:", results)
