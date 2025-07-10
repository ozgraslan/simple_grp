import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoModel, AutoImageProcessor, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from tqdm import tqdm
import wandb

import tyro
from dataclasses import dataclass

@dataclass
class FinetuneConfig:
    save_model_name: str  #= "siglip_finetuned_real" # libero
    repo_id: str #= "mlfu7/pi0_conversion" # "lerobot/libero_10_image" 
    img_key: str #= "exterior_image_1_left" # "observation.images.image"  
    project_name: str = "siglip-finetuning"
    model_text: str = "google/siglip-base-patch16-224"

    batch_size: int = 512
    num_epochs: int = 10
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.05
    seed: int = 42
    use_scheduler: bool = True

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


class LiberoImageTextDataset(Dataset):
    def __init__(self, repo_id, model_text, img_key='exterior_image_1_left'):

        self.metadata = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(repo_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_text)
        self.processor = AutoImageProcessor.from_pretrained(model_text, do_rescale=False)
        self.img_key = img_key

        self.preprocess_task_texts()

    def preprocess_task_texts(self):
        text_ids_list = []
        task_ids = []
        ## did a loop because I am not sure if items returned in order
        tasks = self.metadata.tasks
        for task_id, task_text in tasks.items():
            task_ids.append(task_id)
            print(f"This is a photo of a robot arm tasked to {task_text}")
            token_ids = self.tokenizer(f"This is a photo of a robot arm tasked to {task_text}", padding="max_length", max_length=64, return_tensors="pt").input_ids
            text_ids_list.append(token_ids)

        task_ids = torch.tensor(task_ids, dtype=torch.long)
        ## preserve order using task ids
        text_token_ids = torch.cat(text_ids_list, dim=0)[task_ids]
        self.text_token_ids = text_token_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        imgs = item[self.img_key]
        task_index = item['task_index']
        proc_imgs = self.processor(images=imgs, return_tensors="pt").pixel_values
        token_ids = self.text_token_ids[task_index]    
        return {
            'pixel_values': proc_imgs.squeeze(0),  # Remove batch dimension
            'input_ids': token_ids,
        }

if __name__ == "__main__":        

    cfg = tyro.cli(FinetuneConfig)
    print(cfg)
    set_seed_everywhere(cfg.seed)

    model = AutoModel.from_pretrained(cfg.model_text, attn_implementation="sdpa")
    for param in model.text_model.parameters():
        param.requires_grad = False

    dataset = LiberoImageTextDataset(cfg.repo_id, cfg.model_text, img_key=cfg.img_key)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    accelerator = Accelerator(mixed_precision="bf16")
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Scheduler
    if cfg.use_scheduler:
        total_steps = len(dataloader) * cfg.num_epochs
        warmup_steps = int(cfg.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


    if accelerator.is_main_process:
        wandb.init(project=cfg.project_name, config=vars(cfg))

    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader)):
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            if cfg.use_scheduler:
                scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss})


    # ===========================
    # Save model
    # ===========================
    if accelerator.is_main_process:
        model.save_pretrained(cfg.save_model_name)
        dataset.processor.save_pretrained(cfg.save_model_name)
        wandb.finish()
