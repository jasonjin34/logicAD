"""
This is the help functions for extraction sliding windows
"""
import ipdb
import torch
from einops import rearrange


def generate_sliding_images(
    x: torch.Tensor,
    patch_size: int = 16,
    window_size: int = 240
):
    """
    x [B, 3, 240, 240]
    x [B, 3, 224, 224]
    """
    width, height = x.shape[2:]
    num_patch_width = int(width// patch_size)
    num_patch_height = int(height// patch_size)
    if num_patch_width % 2 or num_patch_height % 2:
        x = x.unfold(2, window_size, window_size).unfold(
            3, window_size, window_size)
    else:
        x = x.unfold(2, window_size, window_size // 2).unfold(
            3, window_size, window_size // 2)
    return x

def generate_sliding_windows(
    x: torch.Tensor,
    patch_size: int = 16,
    stride: int = 112,
    window_size: int = 224
):
    """
    Using Unfold for cutting in the input image into equal patches
    Args:
        x: torch.Tensor input image, ie.e x [B, 3, 224, 224]
        stride: for moving image
        window_size: sliding windows size

    Examples:
        x =
        [1, 2, 3, 4, 5]
        [6, 7, 8, 9, 10]
        [11, 12, 13, 14, 15]
        [16, 17, 18, 19, 20]
        [21, 22, 23, 24, 25]
        if the stride is 2 and windows is 3, the result cutted image will be

        [1, 2, 3]           [3, 4, 5]    ...
        [6, 7, 8]           [8, 9, 10]   ...
        [11, 12, 13]        [13, 14, 15] ...
    """
    assert (len(x.shape) == 4)
    img_width, img_height = x.shape[2:]

    num_patch_width = int(img_width // patch_size)
    num_patch_height = int(img_height // patch_size)

    patch_index_map = torch.arange(
        num_patch_width * num_patch_width).reshape(num_patch_width, num_patch_height)

    # mask for each sliding window
    if stride % patch_size == 0: # 112 % 16 == 0
        stride = stride // patch_size
    else: # 120 % 16 != 0 
        stride = 2 * stride // patch_size

    window_size = window_size // patch_size
    mask = patch_index_map.unfold(0, window_size, stride).unfold(1, window_size, stride)
    mask = rearrange(mask, "h w i j -> (h w) (i j)")
    weights = mask.unique(return_counts=True)[1].float()  # avoid type cast error
    mask = mask.to(x.device)
    weights = weights.to(x.device)

    output = {
        "mask": mask,
        "weights": weights
    }
    return output


if __name__ == "__main__":
    x = torch.rand(4, 3, 448, 448)
    output = generate_sliding_windows(x)
    print("test")



