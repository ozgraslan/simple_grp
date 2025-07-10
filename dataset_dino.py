import numpy as np
import torch
from torch.utils.data import DataLoader
from tensordict import TensorDict, MemoryMappedTensor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import timm
from transformers import AutoImageProcessor

from tqdm import tqdm

import psutil
import os



@torch.no_grad()
def encode_img(b_img):
    transformed_img = img_transform((255 * b_img).to(torch.uint8).permute(0,2,3,1).cpu().numpy()).to(device)
    output = dino.forward_features(transformed_img)
    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
    return output


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # dtype = torch.bfloat16 if transformers.file_utils.is_torch_bf16_available() else torch.float32

    repo_id = "lerobot/libero_spatial_image"

    selected_columns = {"observation.images.image": "image", "task_index": "task_index",
                        "observation.state": "state", "action": "action", "action_is_pad": "valid_mask"}

    num_action_tokens = 32


    data_dct = {"image": {"shape": (257, 384), "dtype": dtype},
                "state": {"shape": (8,), "dtype": dtype},
                "action": {"shape": (num_action_tokens, 7), "dtype": dtype},
                "action_is_pad" : {"shape": (num_action_tokens,), "dtype": dtype},
                "task_index": {"shape": (), "dtype": dtype},
    }

    img_size = 224
    dino = timm.create_model('timm/vit_small_patch14_dinov2.lvd142m',
                            pretrained=True,    
                            img_size=img_size,
                            num_classes=0,  # remove classifier nn.Linear
    ).to(device)
    dino.eval()
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', size={"shortest_edge": img_size})
    img_transform = lambda imgs: processor(images=imgs, return_tensors="pt").pixel_values
    save_path = "/network/scratch/o/ozgur.aslan/libero_td/libero_sp_dino_224_dataset"



    dataset_metadata = LeRobotDatasetMetadata(repo_id)
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
                out = encode_img(b_img)
                dct[selected_columns[key]] = out #.to(device, dtype=dtype)
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
            train_td_dict[start:finish].save(f"{save_path}_{i+1}.pt")
            train_td_dict.clear()
            train_td_dict = get_empty_td_dataset(data_dct, bs*(len(dataset)//bs), device="cpu")
            start = (i+1)*bs
    train_td_dict[start:].save(f"{save_path}_{i+1}.pt")
