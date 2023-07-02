# =============================================================================
# Import required libraries
# =============================================================================
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def imshow(image_tensor):
    image_tensor = (image_tensor / 2) + 0.5
    image_tensor = image_tensor.detach().cpu().numpy()
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    image = image_tensor.transpose(1, 2, 0)
    image = image.clip(0, 1)
    plt.imshow(image)


def grid_imshow(image_tensor, timestep=None, num_images=16):
    image_tensor = (image_tensor / 2) + 0.5
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    image_grid = image_grid.permute(1, 2, 0).squeeze()
    image_grid = image_grid.clip(0, 1)
    plt.imshow(image_grid)
    if timestep != None:
        plt.title("Time Steps: " + str(timestep))
    plt.show()
