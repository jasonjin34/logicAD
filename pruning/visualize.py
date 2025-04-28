import matplotlib.pyplot as plt

def visualize_head_mask(head_mask, title="Attention Head Mask (Black = Pruned, White = Kept)"):
    """
    Visualize the attention head mask.
    Black = Pruned (0), White = Kept (1)
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(head_mask, cmap="Greys_r", aspect="auto")  # inverted color map
    plt.colorbar(label="1 = Kept | 0 = Pruned")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    plt.show()
