import torch
import timm
from transformers import AutoImageProcessor

from base_feature_extractor import BaseFeatureExtractor
from utils import is_uint8

class Dinov2Inference(BaseFeatureExtractor):
    def __init__(self, config):
        self.model =  timm.create_model(
            config.model_name,
            pretrained=True,    
            img_size=config.img_size,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.model_device = next(self.model.parameters()).device
        processor = AutoImageProcessor.from_pretrained(config.model_name, size={"shortest_edge": config.img_size})
        self.img_transform = lambda imgs, do_rescale: processor(images=imgs, do_rescale=do_rescale, return_tensors="pt").pixel_values

    @torch.no_grad()
    def embed(self, images, **kwargs):
        transformed_img = self.img_transform(images, do_rescale = is_uint8(images))
        output = self.model.forward_features(transformed_img)
        return {"image": output}
    
    def prepare_tasks(self, **kwargs):
        return None

    def to(self, device_or_dtype):
        self.model = self.model.to(device_or_dtype)
        self.model_device = next(self.model.parameters()).device

    @property
    def device(self):
        return self.model_device

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
