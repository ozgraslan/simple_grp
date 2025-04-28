import numpy as np
import torch
from torchvision import transforms

from PIL import Image
import timm

from visualize_features import visualize_features_lch

img_size = 224
img_transform = transforms.Compose(
    [   
        transforms.Lambda(lambda img: img if isinstance(img, torch.Tensor) else torch.from_numpy(img) if isinstance(img, np.ndarray) else torch.from_numpy(np.array(img))),
        transforms.Lambda(lambda img: img.permute(0,3,1,2) if len(img.shape) == 4 else img.permute(2,0,1)),
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(
            img_size,
        ),

        transforms.Lambda(lambda img: img.div(255) if isinstance(img, torch.ByteTensor) else img),
        transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
        transforms.Lambda(lambda img: img.unsqueeze(0) if len(img.shape) == 3 else img),
    ]
)
if __name__ == "__main__":
    img = Image.open("../data/dog.jpeg")

    model = timm.create_model('timm/vit_small_patch14_reg4_dinov2.lvd142m', # 'vit_large_patch14_reg4_dinov2.lvd142m', 
                            pretrained=True,    
                            img_size=img_size,
                            num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # print(data_config)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    # print(transforms)
    # exit(0)
    with torch.no_grad():
        output = model.forward_features(img_transform(img))[0, 5:].permute(1,0).reshape(-1, img_size//14, img_size//14)
    print(output.shape)
        
    visualize_features_lch(output, f"dinoreg_small_{img_size}_mt_dog.png")