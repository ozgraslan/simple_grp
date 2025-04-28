import torch

from PIL import Image
import timm

from visualize_features import visualize_features_lch


if __name__ == "__main__":
    img = Image.open("./data/agentview_image.png")

    model = timm.create_model('sam2_hiera_base_plus.fb_r896_2pt1', 
                            pretrained=True,    
                            num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    with torch.no_grad():
        output = model.forward_features(transforms(img).unsqueeze(0))[0].permute(2,0,1)
    
    print(output.shape)
    visualize_features_lch(output, "sam_agentview_image.png")