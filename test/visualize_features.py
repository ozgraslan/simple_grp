import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from skimage import color

from torchvision.transforms.functional import gaussian_blur



def gaussian_blur_torch(x, kernel_size=3, sigma=1):
    return gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)  # shape (C, H, W)

def pca_to_lch(pca_result):
    pca_norm = (pca_result - pca_result.min(0)) / (pca_result.max(0) - pca_result.min(0))
    L = 100 * pca_norm[..., 0]
    C = 100 * pca_norm[..., 1]
    h = 360 * pca_norm[..., 2]
    return np.stack([L, C, h], axis=-1)

def visualize_features_lch(x_tensor, save_path="test.png"):
    # x_tensor: torch tensor of shape (C, H, W)
    
    x_blur_tensor = gaussian_blur_torch(x_tensor)
    x_avg = 0.5 * x_tensor + 0.5 * x_blur_tensor  # shape (C, H, W)

    H, W = x_tensor.shape[1], x_tensor.shape[2]
    x_flat = x_avg.reshape(x_tensor.shape[0], -1).T  # shape (H*W, C)

    x_flat = x_flat.numpy()
    # PCA to 3D
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_flat).reshape(H, W, 3)

    # Map to LCh then to RGB
    lch = pca_to_lch(x_pca)
    lab = color.lch2lab(lch)
    rgb = color.lab2rgb(lab)

    # Visualize
    plt.imshow(rgb)
    plt.title("LCh PCA Visualization")
    plt.axis("off")
    plt.savefig(save_path)
