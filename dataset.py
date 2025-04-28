import torch
from torch.utils.data import DataLoader
from tensordict import TensorDict, MemoryMappedTensor

import transformers
from transformers import AutoProcessor, AutoBackbone, AutoConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from ctrlo.inference import CTRLOFeatureExtractor
from torch.cuda.amp import autocast

from tqdm import tqdm

import psutil
import os


@torch.no_grad()
def encode_img(img, vis_config, vis_processor, vis_backbone):
    processed = vis_processor(images=(255 * img).to(torch.uint8).permute(0,2,3,1).cpu().numpy(), return_tensors="pt").to(vis_backbone.device)
    img_embeds = vis_backbone(**processed, output_hidden_states=True, return_dict=True).hidden_states[-1]
    torch.cuda.empty_cache()  # Clears cache # without this model leads to reserving too much memory and crashing in the next iter
    return img_embeds[:, vis_config.num_register_tokens+1:]

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
    dtype = torch.bfloat16 if transformers.file_utils.is_torch_bf16_available() else torch.float32

    repo_id = "lerobot/libero_10_image"
    vis_name = "facebook/dinov2-with-registers-base"

    selected_columns = {"observation.images.image": "image", "task_index": "task_index",
                        "observation.state": "state", "action": "action", "action_is_pad": "valid_mask"}

    num_action_tokens = 32

        
    vis_config = AutoConfig.from_pretrained(vis_name, out_features=[], torch_dtype=dtype)
    vis_backbone = AutoBackbone.from_pretrained(vis_name, out_features=[], torch_dtype=dtype).to("cuda")
    vis_backbone.eval()
    vis_processor = AutoProcessor.from_pretrained(vis_name, out_features=[], torch_dtype=dtype)

    data_dct = {"image": {"shape": (vis_config.num_register_tokens+1, vis_config.hidden_size), "dtype": dtype},
                "state": {"shape": (8,), "dtype": dtype},
                "action": {"shape": (num_action_tokens, 7), "dtype": dtype},
                "action_is_pad" : {"shape": (num_action_tokens,), "dtype": dtype},
                "task_index": {"shape": (), "dtype": dtype},
    }

    dataset_metadata = LeRobotDatasetMetadata(repo_id)
    ds_features = dataset_metadata.features

    delta_timestamps = {
        "action": [t / dataset_metadata.fps for t in range(num_action_tokens)],
    }

    bs = 128

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    # print(dataset["action"])
    print(f"Number of frames in training dataset (100s% subset): {len(dataset)}")

    # td_dataset = TensorDict.load_memmap("/network/projects/real-g-grp/libero10_dinov2_base_patch_dataset_791.pt").to(device)
    # print(td_dataset)
    # td_dataset = TensorDict.load_memmap("/network/projects/real-g-grp/libero10_dinov2_base_patch_dataset_488.pt").to(device)
    # print(td_dataset)

    # # val_dataset = TensorDict.load_memmap("/network/projects/real-g-grp/libero10_dinov2reg_val_td.pt").to(device)

    # exit(0)
    loader = DataLoader(dataset,
                          num_workers=4,
                          batch_size=bs,
                          shuffle=False,
                          pin_memory=device != "cpu",
                          drop_last=True,) # makes avg comp easier
    for batch in loader:
        print(type(batch["task"]))
        exit(0)

    train_td_dict = get_empty_td_dataset(data_dct, bs*(len(dataset)//bs), device="cpu")
    print(train_td_dict)
    start = 0 
    finish = 0
    for i, batch in enumerate(tqdm(loader)):
        dct = {}
        for key in selected_columns.keys():
            if key == "observation.images.image":
                dct[selected_columns[key]] = encode_img(batch[key]).to(device, dtype=dtype)
            elif key == "action_is_pad":
                dct[selected_columns[key]] = (~ batch[key]).to(device=device, dtype=dtype)
            else:
                dct[selected_columns[key]] = batch[key].to(device=device, dtype=dtype)

        train_td_dict[i*bs:(i+1)*bs] = TensorDict(dct)
        mem_in_mb = process.memory_info().rss / 1024 ** 2
        if i % 100 == 0:
            print(f"Memory usage: {mem_in_mb:.2f} MB")
        if mem_in_mb > 30000:
            finish = (i+1)*bs
            print(train_td_dict[start:finish])
            train_td_dict[start:finish].save(f"/network/projects/real-g-grp/libero10_dinov2_base_reg_dataset_{i+1}.pt")
            train_td_dict.clear()
            train_td_dict = get_empty_td_dataset(data_dct, bs*(len(dataset)//bs), device="cpu")
            start = (i+1)*bs
    train_td_dict[start:].save(f"/network/projects/real-g-grp/libero10_dinov2_base_reg_dataset_{i+1}.pt")
