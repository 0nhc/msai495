from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path:str,
               img_format:str = 'L'):
    """
    Load an image from the specified path and convert it to a numpy array.
    
    Args:
        image_path (str): The path to the image file.
        output_format (str): The desired output format for the image array.
                             Default is 'uint8'.
    
    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    img = Image.open(image_path)
    # img = img.convert(img_format)
    img = np.array(img)
    return img

def display_image(image_array:np.ndarray):
    """
    Display an image using matplotlib.
    
    Args:
        image_array (np.ndarray): The image array to display.
    """
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.show()


path1 = os.path.join(os.path.dirname(__file__), 'dataset', 'pointer1_rgb.png')
path2 = os.path.join(os.path.dirname(__file__), 'dataset', 'pointer1_mask.png')
# path3 = os.path.join(os.path.dirname(__file__), 'results', 'pointer1_masked_nrgb.png')
# path4 = os.path.join(os.path.dirname(__file__), 'results', 'pointer1_masked_hsi.png')

img1 = load_image(path1)
img2 = load_image(path2)
# img3 = load_image(path3)
# img4 = load_image(path4)

# merge the images into 1x4 with matplotlib
fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
axs[0].imshow(img1)
axs[0].set_title('Original Image')
axs[1].imshow(img2, cmap='gray')
axs[1].set_title('Ground Truth Mask from SAM 2')
# axs[2].imshow(img3)
# axs[2].set_title('Masked with nRGB')
# axs[3].imshow(img4)
# axs[3].set_title('Masked with HSI')
for ax in axs:
    ax.axis('off')  # Hide axes
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'pointer1_data.png'), bbox_inches='tight', dpi=300)