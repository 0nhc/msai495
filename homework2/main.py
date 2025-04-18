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
    img = img.convert(img_format)
    img = img.point(lambda x: x // 255)
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

def erosion(img: np.ndarray,
            ksize: int):
    """
    Binary erosion with a square structuring element of ones of size ksize*ksize.

    Args:
        img (np.ndarray): 2D binary array (values 0 or 1).
        ksize (int): Size of the square structuring element (must be >=1).

    Returns:
        np.ndarray: Eroded binary image (0 or 1), same shape as img.
    """
    pad = ksize // 2
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            # assume erosion succeeds until we find a zero under the SE
            erode_pixel = 1
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    y, x = i + di, j + dj
                    # outside image bounds or background pixel → erosion fails
                    if not (0 <= y < H and 0 <= x < W and img[y, x] == 1):
                        erode_pixel = 0
                        break
                if erode_pixel == 0:
                    break
            out[i, j] = erode_pixel
    return out

def dilation(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Binary dilation with a square structuring element of 1’s of size ksize×ksize.

    Args:
        img (np.ndarray): 2D binary array (values 0 or 1).
        ksize (int): Size of the square structuring element (must be >=1).

    Returns:
        np.ndarray: Dilated binary image (0 or 1), same shape as img.
    """
    if ksize < 1:
        raise ValueError("ksize must be at least 1")
    pad = ksize // 2
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            dilate_pixel = 0
            # if any neighbor under the SE is foreground, set output to 1
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    y, x = i + di, j + dj
                    if 0 <= y < H and 0 <= x < W and img[y, x] == 1:
                        dilate_pixel = 1
                        break
                if dilate_pixel:
                    break
            out[i, j] = dilate_pixel
    return out

def opening(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Morphological opening: erosion followed by dilation.

    Args:
        img (np.ndarray): 2D binary array (0 or 1).
        ksize (int): Size of the square structuring element.

    Returns:
        np.ndarray: Opened image.
    """
    eroded = erosion(img, ksize)
    opened = dilation(eroded, ksize)
    return opened

def closing(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Morphological closing: dilation followed by erosion.

    Args:
        img (np.ndarray): 2D binary array (0 or 1).
        ksize (int): Size of the square structuring element.

    Returns:
        np.ndarray: Closed image.
    """
    dilated = dilation(img, ksize)
    closed = erosion(dilated, ksize)
    return closed

def boundary(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Boundary extraction: original minus its erosion.

    Args:
        img (np.ndarray): 2D binary array (0 or 1).
        ksize (int): Size of the square structuring element.

    Returns:
        np.ndarray: Boundary (1 where pixel is removed by erosion).
    """
    eroded = erosion(img, ksize)
    boundary = img.astype(int) - eroded.astype(int)
    return (boundary > 0).astype(np.uint8)

def main():
    """
    Main function.
    """
    # Load and display the original image
    path = os.path.join(os.path.dirname(__file__), 'instructions', 'palm.bmp')
    img = load_image(path)
    display_image(img)

    # Perform morphological operations
    erosion_img = erosion(img, 3)
    dilation_img = dilation(img, 3)
    boundary_img = boundary(img, 3)
    opening_img = opening(img, 3)
    closing_img = closing(img, 3)

    # Display the results
    display_image(erosion_img)
    display_image(dilation_img)
    display_image(boundary_img)
    display_image(opening_img)
    display_image(closing_img)

if __name__ == "__main__":
    main()
