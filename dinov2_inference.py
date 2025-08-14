import torch
from torch import nn
import timm
from transformers import AutoImageProcessor

from utils import is_uint8

from dataclasses import dataclass

@dataclass
class Dinov2Config:
    model_name: str = "timm/vit_small_patch14_reg4_dinov2.lvd142m"
    processor_name: str = "facebook/dinov2-small"
    img_size: str = 224

class Dinov2Inference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model =  timm.create_model(
            config.model_name,
            pretrained=True,    
            img_size=config.img_size,
            num_classes=0,  # remove classifier nn.Linear
        )
        processor = AutoImageProcessor.from_pretrained(config.processor_name, size={"shortest_edge": config.img_size})
        self.img_transform = lambda imgs, do_rescale: processor(images=imgs, do_rescale=do_rescale, return_tensors="pt").pixel_values

    def forward(self, images):
        transformed_img = self.img_transform(images, do_rescale = is_uint8(images))
        output = self.model.forward_features(transformed_img)
        return output

    @torch.no_grad()
    def embed(self, images, **kwargs):
        return {"image": self.forward(images)}
    
    def prepare_tasks(self, **kwargs):
        return None

    @property
    def device(self):
        return next(self.parameters()).device
    

if __name__ == "__main__":
    config = Dinov2Config()
    dinov2 = Dinov2Inference(config)
    zero_images = torch.zeros((10,512,512,3), dtype=torch.uint8)

    print(dinov2(zero_images))