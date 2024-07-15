from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from einops import rearrange


# load image and transform it to tensor
def load_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((600, 300))
    image = torch.tensor(np.asanyarray(image)).permute(2, 0, 1)
    return image


def display_image(image: torch.tensor):
    img = image.permute(1, 2, 0).numpy()
    plt.imshow(img)


def image_unfold(
    image: torch.tensor,
    patch_size: int = 400,
    stride: float = 0.5,
    pos: int = 1,
):
    image = image.unfold(1, patch_size, int(patch_size * stride)).unfold(2, patch_size, int(patch_size * stride))
    image = rearrange(image, "c i j h w -> (i j) c h w")
    if pos > image.shape[0]:
        raise ValueError(f"Position {pos} is out of range, we have only {image.size[0]} patches")
    if pos == -1:
        return image
    else:
        return image[pos]


def display_multiple_image(listimage):
    fig, axs = plt.subplots(len(listimage), 1, figsize=(20, 10))
    for i, img in enumerate(listimage):
        axs[i].imshow(img.permute(1, 2, 0).numpy())
        axs[i].axis("off")
    plt.show()
