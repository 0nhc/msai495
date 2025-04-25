from PIL import Image
import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt

def load_image(image_path:str,
               img_format:str = 'rgb'):
    """
    Load an image from the specified path and convert it to a numpy array.
    
    Args:
        image_path (str): The path to the image file.
        img_format (str): The desired output format for the image array.
                             'rgb' or 'grasyscale'. Default is 'rgb'.
    
    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    img = Image.open(image_path)
    if img_format == 'rgb':
        img = img.convert('RGB')
    elif img_format == 'grayscale':
        img = img.convert('L')
    # resize the image to 640x480
    # img = img.resize((640, 480))
    img = np.array(img)
    return img

def display_image(image_array: np.ndarray, title: str = None):
    if title is None:
        title = 'Image'

    # Convert RGB→BGR if needed
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # create a resizable window
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # show it
    cv2.imshow(title, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize_nonzero(hist: np.ndarray) -> np.ndarray:
    """
    Linearly scale all non-zero entries of `hist` so that the maximum
    non-zero becomes 1. Zeros remain zero.
    """
    # find the maximum value among the non-zero bins
    nonzero = hist[hist > 0]
    if nonzero.size == 0:
        return hist  # nothing to do
    
    max_val = nonzero.max()
    # divide only the non-zero entries by max_val
    hist_norm = hist.copy()
    hist_norm[hist_norm > 0] /= max_val
    return hist_norm

def compute_rg_histogram(dataset_dir: str,
                         color_space:str = 'rgb') -> np.ndarray:
    """
    Compute a 2D histogram over R and G values across all images,
    counting only pixels where the corresponding mask >0.

    Args:
        dataset_dir (str): Path to the root of your dataset (contains subfolders).

    Returns:
        hist (np.ndarray): 256×256 array where hist[r, g] is the probability
                           of seeing R=r and G=g in the masked regions.
    """
    hist = np.zeros((256, 256), dtype=np.float64)
    total_pixels = 0

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.endswith('rgb.png'):
                continue
            rgb_path = os.path.join(root, fname)
            mask_path = rgb_path.replace('rgb.png', 'mask.png')
            if not os.path.exists(mask_path):
                continue

            rgb = load_image(rgb_path, img_format='rgb')
            if(color_space == 'rgb'):
               img = rgb 
            elif(color_space == 'nrgb'):
                img = rgb_to_nrgb(rgb)
            elif(color_space == 'hsi'):
                img = rgb_to_hsi(rgb)
            else:
                raise ValueError(f"Unknown color space: {color_space}")
            # display_image(img, title='Image')
            mask = load_image(mask_path, img_format='grayscale')

            # Boolean mask of valid pixels
            valid = mask > 0
            if not np.any(valid):
                continue

            # Extract R and G channels where mask is true
            R = img[..., 0][valid].ravel()
            G = img[..., 1][valid].ravel()

            # Accumulate counts
            # np.add.at handles repeated indices efficiently
            np.add.at(hist, (R, G), 1)
            total_pixels += R.size

    # Convert counts to probabilities
    if total_pixels > 0:
        hist /= total_pixels

    return hist

def visualize_histogram(hist: np.ndarray):
    """
    Display a 256×256 float histogram as a grayscale image using PIL.

    Args:
        hist (np.ndarray): 256×256 array of probabilities.
    """
    # Scale to [0,255] for display
    disp = (hist * 255).clip(0, 255).astype(np.uint8)

    # Create a PIL Image in 'L' mode (8-bit pixels, black and white)
    img = Image.fromarray(disp, mode='L')
    img.show()  # opens the default image viewer

def segment_by_histogram(img: np.ndarray,
                         hist: np.ndarray,
                         threshold: float = 0.1,
                         normalize: bool = True
                        ) -> np.ndarray:
    """
    Segment an RGB image by looking up each pixel's (R,G) in a 2D histogram.

    Args:
        img (np.ndarray): H×W×3 RGB image, values in [0..255].
        hist (np.ndarray): 256×256 float array of probabilities for each (r,g).
        threshold (float): cut‐off in [0..1]; pixels with prob > threshold become 255.
        normalize (bool): if True, first scale non-zero hist bins so max → 1.

    Returns:
        mask (np.ndarray): H×W uint8 array, 255 where hist[r,g]>threshold, else 0.
    """
    # 1) optionally normalize non-zero entries so the largest bin → 1
    if normalize:
        nz = hist > 0
        if np.any(nz):
            hist = hist.copy()
            hist[nz] /= hist[nz].max()

    # 2) look up per-pixel probability
    R = img[..., 0].astype(np.uint8)
    G = img[..., 1].astype(np.uint8)
    prob_map = hist[R, G]  # shape H×W

    # 3) threshold → binary mask (0 or 255)
    mask = (prob_map > threshold).astype(np.uint8) * 255
    return mask

def rgb_to_nrgb(rgb: np.ndarray,
                eps: float = 1e-12):
    """
    Convert an RGB image to an nRGB image.

    Args:
        rgb (np.ndarray): HxWx3 RGB image, values in [0-255].
        eps (float): Small value to avoid division by zero.

    Returns:
        nRGB (np.ndarray): HxWx3  RGB image, values in [0-255].
    """
    # Normalize the RGB values to [0..1]
    sum_rgb = np.sum(rgb, axis=2, keepdims=True) + eps
    nrgb = rgb / sum_rgb
    # rescale to 0-255 for each channel
    nrgb = nrgb * 3 * 255
    # Clip values to be in the range [0, 255]
    nrgb = np.clip(nrgb, 0, 255).astype(np.uint8)
    return nrgb

def rgb_to_hsi(rgb: np.ndarray):
    """
    Convert an RGB image to an HSI image.

    Args:
        rgb (np.ndarray): HxWx3 RGB image, values in [0-255].

    Returns:
        hsi (np.ndarray): HxWx3 HSI image, values in [0-255].
    """
    # Normalize the RGB values to [0..1]
    rgb = rgb.astype(np.float32) / 255.0
    # Compute the intensity
    I = np.mean(rgb, axis=2)
    # Compute the saturation
    min_rgb = np.min(rgb, axis=2)
    S = 1 - (min_rgb / (I + 1e-10))
    # Compute the hue
    num = 0.5 * ((rgb[..., 0] - rgb[..., 1]) + (rgb[..., 0] - rgb[..., 2]))
    denom = np.sqrt((rgb[..., 0] - rgb[..., 1])**2 + (rgb[..., 0] - rgb[..., 2]) * (rgb[..., 1] - rgb[..., 2]))
    theta = np.arccos(num / (denom + 1e-10))
    H = np.zeros_like(I)
    H[rgb[..., 2] <= rgb[..., 1]] = theta[rgb[..., 2] <= rgb[..., 1]]
    H[rgb[..., 2] > rgb[..., 1]] = (2 * np.pi) - theta[rgb[..., 2] > rgb[..., 1]]
    
    # Stack the channels to form the HSI image
    hsi = np.stack((H, S, I), axis=-1)

    # Scale the HSI values to [0, 255]
    hsi[..., 0] = (hsi[..., 0] / (2 * np.pi)) * 255  # Hue
    hsi[..., 1] = (hsi[..., 1]) * 255  # Saturation
    # Clip values to be in the range [0, 255]
    hsi = np.clip(hsi, 0, 255).astype(np.uint8)
    return hsi


def main():
    """
    Main function.
    """
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    # test_rgb_path = os.path.join(os.path.dirname(__file__), 'instructions', 'gun1.bmp')
    # test_rgb_path = os.path.join(os.path.dirname(__file__), 'instructions', 'joy1.bmp')
    # test_rgb_path = os.path.join(os.path.dirname(__file__), 'instructions', 'pointer1.bmp')
    test_rgb_path = os.path.join(os.path.dirname(__file__), 'dataset', 'middle_finger_rgb.png')

    """
    RGB Histogram
    """
    # # Compute the histogram over all masked pixels
    # hist2d = compute_rg_histogram(dataset_dir)
    # # Normalize the histogram
    # hist2d = normalize_nonzero(hist2d)
    # # Visualize it
    # visualize_histogram(hist2d)
    # # Test the histogram on a single image
    # rgb_img = load_image(test_rgb_path, img_format='rgb')
    # rgb_to_nrgb(rgb_img)
    # mask = segment_by_histogram(rgb_img, hist2d, threshold=1e-3)
    # # mask the image
    # img_masked = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    # # Display the original image and the mask
    # display_image(img_masked, title='Original Image')

    """
    nRGB Histogram
    """
    # # Compute the histogram over all masked pixels
    # hist2d = compute_rg_histogram(dataset_dir, color_space='nrgb')
    # # Normalize the histogram
    # hist2d = normalize_nonzero(hist2d)
    # # Visualize it
    # visualize_histogram(hist2d)
    # # Test the histogram on a single image
    # rgb_img = load_image(test_rgb_path, img_format='rgb')
    # nrgb_img = rgb_to_nrgb(rgb_img)
    # mask = segment_by_histogram(nrgb_img, hist2d, threshold=0.25)
    # # mask the image
    # img_masked = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    # # Display the original image and the mask
    # display_image(img_masked, title='Original Image')

    """
    HSI Histogram
    """
    # # Compute the histogram over all masked pixels
    # hist2d = compute_rg_histogram(dataset_dir, color_space='hsi')
    # # Normalize the histogram
    # hist2d = normalize_nonzero(hist2d)
    # # Visualize it
    # visualize_histogram(hist2d)
    # # Test the histogram on a single image
    # rgb_img = load_image(test_rgb_path, img_format='rgb')
    # hsi_img = rgb_to_hsi(rgb_img)
    # mask = segment_by_histogram(hsi_img, hist2d, threshold=1e-2)
    # # mask the image
    # img_masked = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    # # Display the original image and the mask
    # display_image(img_masked, title='Original Image')

if __name__ == "__main__":
    main()
