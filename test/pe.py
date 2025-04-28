import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from visualize_features import visualize_features_lch

# print("CLIP configs:", pe.CLIP.available_configs())
# # CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

# model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)  # Downloads from HF
# model = model.cuda()

# preprocess = transforms.get_image_transform(model.image_size)
# tokenizer = transforms.get_text_tokenizer(model.context_length)

# image = preprocess(Image.open("./data/agentview_image.png")).unsqueeze(0).cuda()
# text = tokenizer(["a basket", "a robot arm", "a soup can", "an orange juice", "a dog", "a cat", "put both the alphabet soup and the cream cheese box in the basket"]).cuda()

# with torch.no_grad(), torch.autocast("cuda"):
#     image_features, text_features, logit_scale = model(image, text)
#     text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
#     print(image_features.shape, text_features.shape)

# print("Label probs:", text_probs)  # prints: [[0.0, 0.0, 1.0]]


print("PE configs:", pe.VisionTransformer.available_configs())
# PE configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224', 'PE-Lang-G14-448', 'PE-Lang-L14-448', 'PE-Spatial-G14-448']

model = pe.VisionTransformer.from_config("PE-Core-G14-448", pretrained=True)  # Loads from HF
model = model.cuda()

preprocess = transforms.get_image_transform(model.image_size)
image = preprocess(Image.open("./data/agentview_image.png")).unsqueeze(0).cuda()

with torch.no_grad():

    out = model.forward_features(image)  # pass layer_idx=<idx> to get a specific layer's output!

print(out.shape)
out = out[0].permute(1,0).reshape(-1, 448//14, 448//14).cpu()

visualize_features_lch(out, "pe_core_G_agentview_image.png" )
