import torch
from torch.utils.data import DataLoader
from tensordict import TensorDict, MemoryMappedTensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

import psutil
import os

from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, Optional


@dataclass
class Config:
    # General settings
    model_name: str
    dataset_repo_id: str
    save_path: str
    task2obj: Optional[Dict]

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    num_action_tokens: int = 32
    batch_size: int = 128
    memory_limit_in_mb: int = 62000

    selected_columns: Dict[str, Union[str, Tuple[str, str]]] = field(default_factory=lambda: {
        "observation.images.image": "image",
        "task_index": "task_index",
        "observation.state": "state",
        "action": "action",
        "action_is_pad": "valid_mask"
    })

    selected_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "image": (),
        "state": (),
        "action": (),
        "task_index": (),
    })


    def make_data_dct(self) -> Dict[str, Dict[str, Union[Tuple[int, ...], torch.dtype]]]:
        data_dct = {key: {"shape": self.selected_shapes[key], "dtype": self.dtype} for key in self.selected_shapes}
        if "action" in data_dct:
            data_dct["action"]["shape"] = (self.num_action_tokens, *self.selected_shapes["action"])
            data_dct["action_is_pad"] = {"shape": (self.num_action_tokens, ), "dtype": self.dtype}
        return data_dct

def get_empty_td_dataset(dct, dataset_size, device="cpu"):
    data_td = TensorDict(
        {key: MemoryMappedTensor.empty((dataset_size, *value["shape"]), dtype=value["dtype"]) for key, value in dct.items()},
        batch_size=[dataset_size],
        device=device,
    )
    return data_td

def process_keys(model, batch, selected_columns, **kwargs):
    dct = {}
    for key in selected_columns:
        if key == "observation.images.image":
            images = batch[key]
            kwargs["task_index"] = batch["task_index"].long()
            out = model.embed(images, **kwargs)
            for new_key in selected_columns[key]:
                dct[new_key] = out[new_key] 
                dct[new_key] = out[new_key]
        elif key == "action_is_pad":
            dct[selected_columns[key]] = (~ batch[key])
        else:
            dct[selected_columns[key]] = batch[key]
    return dct



@torch.no_grad()
def create_td_dataset(model, config):
    process = psutil.Process(os.getpid())
    data_dct = config.make_data_dct()

    dataset_metadata = LeRobotDatasetMetadata(config.dataset_repo_id)

    tasks = model.prepare_tasks(tasks = dataset_metadata.tasks, task2obj = config.task2obj)

    dataset = LeRobotDataset(
        config.dataset_repo_id, 
        delta_timestamps={
        "action": [t / dataset_metadata.fps for t in range(config.num_action_tokens)],
    })

    loader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=config.device != "cpu",
        drop_last=False,
    )

    train_td_dict = get_empty_td_dataset(data_dct, len(dataset), device="cpu")

    index, start = 0, 0
    for batch in tqdm(loader):
        bs = len(batch["task_index"])

        dct = process_keys(model, batch, config.selected_columns, **tasks)
        train_td_dict[index:index+bs] = TensorDict(dct, batch_size=[bs])
        index += bs
        mem_in_mb = process.memory_info().rss / 1024 ** 2
        if (index // config.batch_size) % 100 == 0:
            print(f"Memory usage: {mem_in_mb:.2f} MB")
        if mem_in_mb > config.memory_limit_in_mb:
            train_td_dict[start:index].save(f"{config.save_path}_{index}.pt")
            train_td_dict.clear()
            train_td_dict = get_empty_td_dataset(data_dct, len(dataset), device="cpu")
            start = index
    train_td_dict[start:index].save(f"{config.save_path}_{index}.pt")


