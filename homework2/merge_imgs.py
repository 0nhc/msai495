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


path1 = os.path.join(os.path.dirname(__file__), 'results', 'gun.png')
path2 = os.path.join(os.path.dirname(__file__), 'results', 'gun_dilation.png')
path3 = os.path.join(os.path.dirname(__file__), 'results', 'gun_erosion.png')
path4 = os.path.join(os.path.dirname(__file__), 'results', 'gun_opening.png')
path5 = os.path.join(os.path.dirname(__file__), 'results', 'gun_closing.png')
path6 = os.path.join(os.path.dirname(__file__), 'results', 'gun_boundary.png')

img1 = load_image(path1)
img2 = load_image(path2)
img3 = load_image(path3)
img4 = load_image(path4)
img5 = load_image(path5)
img6 = load_image(path6)

# merge the images into 1x4 with matplotlib
fig, axs = plt.subplots(1, 6, figsize=(24, 4))
axs[0].imshow(img1)
axs[0].set_title('gun.png')
axs[1].imshow(img2)
axs[1].set_title('gun_dilation.png')
axs[2].imshow(img3)
axs[2].set_title('gun_erosion.png')
axs[3].imshow(img4)
axs[3].set_title('gun_opening.png')
axs[4].imshow(img5)
axs[4].set_title('gun_closing.png')
axs[5].imshow(img6)
axs[5].set_title('gun_boundary.png')
for ax in axs:
    ax.axis('off')  # Hide axes
plt.tight_layout()
plt.show()