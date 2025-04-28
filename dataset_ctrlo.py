import torch
from torch.utils.data import DataLoader
from tensordict import TensorDict, MemoryMappedTensor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from ctrlo.inference import CTRLOFeatureExtractor

from tqdm import tqdm

import psutil
import os

@torch.no_grad()
def get_obj_text_embeds(tasks, task2obj):
    text_embed_list = []
    text_mask_list = []
    task_ids = []

    ## did a loop because I am not sure if items returned in order
    for task_id, task_text in tasks.items():
        obj_text_list = task2obj[task_text]
        task_ids.append(task_id)
        text_embed, text_mask = ctrlo.embed_text(obj_text_list)
        # print(text_embed.shape, text_mask.shape)
        text_embed_list.append(text_embed)
        text_mask_list.append(text_mask)

    task_ids = torch.tensor(task_ids, dtype=torch.long)

    ## preserve order using task ids
    text_embeds = torch.stack(text_embed_list, dim=0)[task_ids]
    text_masks = torch.stack(text_mask_list, dim=0)[task_ids]
    return text_embeds, text_masks


@torch.no_grad()
def encode_img(b_img):
    bsz = b_img.shape[0]
    transformed_img = ctrlo.img_transform((255 * b_img).to(torch.uint8).permute(0,2,3,1).cpu().numpy()).to(ctrlo.device)
    empty_loss_mask = torch.zeros((bsz, 7), dtype=int, device=transformed_img.device)
    embeded_empty_text = torch.ones((bsz, 7, 512), device=transformed_img.device)
    outputs = ctrlo.extract_features_batch(transformed_img, empty_loss_mask, embeded_empty_text)
    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
    return outputs["feature_extractor"].features, outputs["perceptual_grouping"].objects

@torch.no_grad()
def encode_img_text(b_img, b_text_embed, b_text_mask):
    transformed_img = ctrlo.img_transform((255 * b_img).to(torch.uint8).permute(0,2,3,1).cpu().numpy()).to(ctrlo.device)
    outputs = ctrlo.extract_features_batch(transformed_img, b_text_mask, b_text_embed)
    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
    return outputs["feature_extractor"].features, outputs["perceptual_grouping"].objects

def get_empty_td_dataset(dct, dataset_size, device="cpu"):
    data_td = TensorDict(
        {key: MemoryMappedTensor.empty((dataset_size, *value["shape"]), dtype=value["dtype"]) for key, value in dct.items()},
        batch_size=[dataset_size],
        device=device,
    )
    return data_td


if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    # Set the device to GPU if available
    img_only = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # dtype = torch.bfloat16 if transformers.file_utils.is_torch_bf16_available() else torch.float32

    repo_id = "lerobot/libero_10_image"

    selected_columns = {"observation.images.image": ("patch", "slot"), "task_index": "task_index",
                        "observation.state": "state", "action": "action", "action_is_pad": "valid_mask"}

    num_action_tokens = 32


    data_dct = {"patch": {"shape": (256, 384), "dtype": dtype},
                "slot": {"shape": (7, 256), "dtype": dtype},
                "state": {"shape": (8,), "dtype": dtype},
                "action": {"shape": (num_action_tokens, 7), "dtype": dtype},
                "action_is_pad" : {"shape": (num_action_tokens,), "dtype": dtype},
                "task_index": {"shape": (), "dtype": dtype},
    }


    dataset_metadata = LeRobotDatasetMetadata(repo_id)
    tasks = dataset_metadata.tasks
    from task2obj import task2obj
    ctrlo = CTRLOFeatureExtractor().to(device)
    ctrlo.eval()
    if not img_only:
        text_embeds, text_masks = get_obj_text_embeds(tasks, task2obj)
        text_embeds, text_masks = text_embeds.to(device), text_masks.to(device)

    # print(text_embeds.shape , text_masks.shape)

    ds_features = dataset_metadata.features

    delta_timestamps = {
        "action": [t / dataset_metadata.fps for t in range(num_action_tokens)],
    }

    bs = 128

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    # print(dataset["action"])
    print(f"Number of frames in training dataset (100s% subset): {len(dataset)}")

    loader = DataLoader(dataset,
                          num_workers=0,
                          batch_size=bs,
                          shuffle=False,
                          pin_memory=device != "cpu",
                          drop_last=True,) # makes avg comp easier

    train_td_dict = get_empty_td_dataset(data_dct, bs*(len(dataset)//bs), device="cpu")
    print(train_td_dict)
    start = 0 
    finish = 0
    for i, batch in enumerate(tqdm(loader)):
        dct = {}
        for key in selected_columns.keys():
            if key == "observation.images.image":
                b_img = batch[key]
                if img_only:
                    out = encode_img(b_img)
                else:
                    b_text_embed = text_embeds[batch["task_index"].long()]
                    b_text_mask = text_masks[batch["task_index"].long()]
                    # print(b_img.shape, b_text_embed.shape, b_text_mask.shape)
                    out = encode_img_text(b_img, b_text_embed, b_text_mask)
                dct[selected_columns[key][0]] = out[0] #.to(device, dtype=dtype)
                dct[selected_columns[key][1]] = out[1] #.to(device, dtype=dtype)
            elif key == "action_is_pad":
                dct[selected_columns[key]] = (~ batch[key]) #.to(device=device, dtype=dtype)
            else:
                dct[selected_columns[key]] = batch[key] #.to(device=device, dtype=dtype)

        train_td_dict[i*bs:(i+1)*bs] = TensorDict(dct)
        mem_in_mb = process.memory_info().rss / 1024 ** 2
        if i % 100 == 0:
            print(f"Memory usage: {mem_in_mb:.2f} MB")
        if mem_in_mb > 62000:
            finish = (i+1)*bs
            print(train_td_dict[start:finish])
            train_td_dict[start:finish].save(f"/network/projects/real-g-grp/libero_td/libero10_ctrlo_img_cc_dataset_{i+1}.pt")
            train_td_dict.clear()
            train_td_dict = get_empty_td_dataset(data_dct, bs*(len(dataset)//bs), device="cpu")
            start = (i+1)*bs
    train_td_dict[start:].save(f"/network/projects/real-g-grp/libero_td/libero10_ctrlo_img_cc_dataset_{i+1}.pt")
