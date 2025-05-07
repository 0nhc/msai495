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

def display_image(image_array: np.ndarray,
                  cmap: str = None,
                  title: str = None):
    if cmap is None:
        plt.imshow(image_array)
    else:
        plt.imshow(image_array, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()

def _make_gauss_kernel(N: int, sigma: float) -> np.ndarray:
    """
    Build a normalized 2D Gaussian kernel.
    
    Args:
        N (int): kernel size (must be odd).
        sigma (float): standard deviation of the Gaussian.
    
    Returns:
        kernel (np.ndarray): shape (N, N), sums to 1.
    """
    # Create a grid of (x,y) coordinates with center at 0
    ax = np.arange(-N//2 + 1, N//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    # Gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    # Normalize so sum = 1
    kernel /= np.sum(kernel)
    return kernel

def gaussian_smoothing(I: np.ndarray, N: int, sigma: float) -> np.ndarray:
    """
    Apply an NxN Gaussian smoothing to image I with standard deviation sigma.

    Args:
        I (np.ndarray): input image, shape (H, W) or (H, W, C), any numeric dtype.
        N (int): size of the Gaussian kernel (must be odd).
        sigma (float): standard deviation of the Gaussian.

    Returns:
        smoothed (np.ndarray): same shape and dtype as I, Gaussian-blurred.
    """
    # Build kernel
    kernel = _make_gauss_kernel(N, sigma)
    pad = N // 2

    # Ensure float for accumulation
    I_float = I.astype(np.float64)
    # Pad edges by repeating the border values
    if I.ndim == 2:
        I_pad = np.pad(I_float, pad_width=pad, mode='edge')
        H, W = I.shape
        out = np.zeros_like(I_float)
        # Slide kernel over image
        for i in range(H):
            for j in range(W):
                region = I_pad[i:i+N, j:j+N]
                out[i, j] = np.sum(region * kernel)
        return out.astype(I.dtype)

    elif I.ndim == 3:
        H, W, C = I.shape
        out = np.zeros_like(I_float)
        # Apply per‐channel
        for c in range(C):
            out[:, :, c] = gaussian_smoothing(I[:, :, c], N, sigma).astype(np.float64)
        return out.astype(I.dtype)

    else:
        raise ValueError("Unsupported input image dimensions: must be HxW or HxWxC")

def image_gradient(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel operators.

    Args:
        S (np.ndarray): input image, shape (H, W) or (H, W, C).
                        If 3-channel, will be converted to grayscale.

    Returns:
        magnitude (np.ndarray): gradient magnitude, shape (H, W), dtype float64
        theta (np.ndarray): gradient orientation in degrees [0,180), shape (H, W), dtype float64
    """
    # 1) If RGB, convert to gray via luminosity
    if S.ndim == 3:
        # weights 0.299, 0.587, 0.114
        S = (0.299 * S[...,0] + 0.587 * S[...,1] + 0.114 * S[...,2]).astype(np.float64)
    else:
        S = S.astype(np.float64)

    # 2) Sobel kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)

    # 3) Pad image to keep same size (pad by 1)
    pad = 1
    S_pad = np.pad(S, pad_width=pad, mode='edge')
    
    H, W = S.shape
    Gx = np.zeros((H, W), dtype=np.float64)
    Gy = np.zeros((H, W), dtype=np.float64)

    # 4) Convolve
    for i in range(H):
        for j in range(W):
            region = S_pad[i:i+3, j:j+3]
            Gx[i, j] = np.sum(region * Kx)
            Gy[i, j] = np.sum(region * Ky)

    # 5) Magnitude and angle
    magnitude = np.hypot(Gx, Gy) # sqrt(Gx^2 + Gy^2)
    theta = np.arctan2(Gy, Gx)

    # 6) Normalize angle to [0, np.pi)
    theta = (theta + np.pi) % np.pi

    return magnitude, theta

def find_threshold(magnitude: np.ndarray,
                   percentage_non_edge: float,
                   num_bins: int = 256
                  ) -> tuple[float, float]:
    """
    Determine low and high thresholds for Canny via the histogram of gradient magnitudes.
    
    Args:
        magnitude (np.ndarray): 2D gradient-magnitude image.
        percentage_non_edge (float): fraction of pixels to treat as non‐edges (0 < p < 1).
        num_bins (int): number of histogram bins to use (default=256).
    
    Returns:
        T_low (float), T_high (float)
    """
    # 1) Flatten and build histogram over [min, max]
    flat = magnitude.ravel()
    mn, mx = flat.min(), flat.max()
    hist, bin_edges = np.histogram(flat, bins=num_bins, range=(mn, mx))

    # Display histogram
    # plt.hist(flat, bins=num_bins, range=(mn, mx), density=True)
    # plt.title('Histogram of Gradient Magnitudes')
    # plt.xlabel('Magnitude')
    # plt.ylabel('Density')
    # plt.show()
    
    # 2) Cumulative distribution (CDF) normalized to [0,1]
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]
    
    # 3) Find the bin where CDF >= percentage_non_edge
    idx = np.searchsorted(cdf, percentage_non_edge, side='right')
    # clamp to valid index
    idx = min(idx, len(bin_edges)-1)
    
    T_high = bin_edges[idx]
    T_low  = 0.5 * T_high
    
    return T_low, T_high

def nonmaxima_suppress(Gx: np.ndarray,
                       Gy: np.ndarray,
                       Mag: np.ndarray,
                       method: str = 'quantization'
                       ) -> np.ndarray:
    """
    Non-maxima suppression using gradient components directly.

    Args:
      Gx, Gy    -- 2D arrays of the x- and y-derivative at each pixel.
      Mag       -- 2D array of gradient magnitudes.
      method    -- 'quantization' (4-way) or 'interpolation'.

    Returns:
      Mag_sup   -- 2D array, same shape as Mag, with non-maxima zeroed out.
    """
    H, W = Mag.shape
    # pad Mag so that neighbors at the border can be sampled
    Mp = np.pad(Mag, ((1,1),(1,1)), mode='edge')
    out = np.zeros_like(Mag)

    # precompute tan thresholds
    t1 = np.tan(np.pi/8)    # ≈0.414 (22.5°)
    t2 = np.tan(3*np.pi/8)  # ≈2.414 (67.5°)

    def _bilinear(img, r, c):
        """Bilinear sample of padded img at float coords (r,c)."""
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = min(r0+1, img.shape[0]-1), min(c0+1, img.shape[1]-1)
        dr, dc = r - r0, c - c0
        return ((1-dr)*(1-dc)*img[r0, c0] +
                (1-dr)*dc    *img[r0, c1] +
                dr   *(1-dc)*img[r1, c0] +
                dr   *dc    *img[r1, c1])

    for i in range(H):
        for j in range(W):
            m  = Mp[i+1, j+1]
            gx = Gx[i, j]
            gy = Gy[i, j]

            if method == 'quantization':
                # avoid division by zero
                if gx == 0:
                    alpha = np.inf
                else:
                    alpha = abs(gy / gx)

                # decide which of the 4 directions
                if alpha <= t1:
                    # horizontal edge (0°)
                    n1, n2 = Mp[i+1, j], Mp[i+1, j+2]
                elif alpha >= t2:
                    # vertical edge (90°)
                    n1, n2 = Mp[i, j+1], Mp[i+2, j+1]
                elif gx * gy > 0:
                    # 45° diagonal
                    n1, n2 = Mp[i, j+2], Mp[i+2, j]
                else:
                    # 135° diagonal
                    n1, n2 = Mp[i, j], Mp[i+2, j+2]

            elif method == 'interpolation':
                # unit‐vector along gradient
                if m == 0:
                    dy = dx = 0.0
                else:
                    dy = gy / m
                    dx = gx / m
                # sample + and – along (dx,dy)
                n1 = _bilinear(Mp, i+1 + dy, j+1 + dx)
                n2 = _bilinear(Mp, i+1 - dy, j+1 - dx)

            else:
                raise ValueError(f"Unknown method '{method}'")

            # keep local maxima only
            if (m >= n1) and (m >= n2):
                out[i, j] = m
            # else remains zero

    return out

def edge_linking(Mag_low: np.ndarray,
                 Mag_high: np.ndarray,
                 connectivity: int = 4
                ) -> np.ndarray:
    """
    Hysteresis linking with selectable connectivity:
      - connectivity=4 → only N,S,E,W neighbors
      - connectivity=8 → N,S,E,W + diagonals

    Args:
      Mag_low   (bool H×W): candidate edges ≥ T_low
      Mag_high  (bool H×W): strong edges ≥ T_high
      connectivity: 4 or 8

    Returns:
      E (bool H×W): final linked edges
    """
    H, W = Mag_low.shape
    E = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)

    if connectivity == 4:
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        neighs = [(-1,-1),(-1,0),(-1,1),
                  ( 0,-1),        ( 0,1),
                  ( 1,-1),( 1,0),( 1,1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    for i in range(H):
        for j in range(W):
            if Mag_high[i, j] and not visited[i, j]:
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if visited[x, y]:
                        continue
                    visited[x, y] = True
                    if Mag_low[x, y]:
                        E[x, y] = True
                        for dx, dy in neighs:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < H and 0 <= ny < W
                                and not visited[nx, ny]
                                and Mag_low[nx, ny]):
                                stack.append((nx, ny))
    return E

def edge(img: np.ndarray,
         N: int = 5,
         sigma: float = 1.0,
         percentage_non_edge: float = 0.75
        ) -> np.ndarray:

    # Apply Gaussian smoothing
    gaussian_smoothing_img = gaussian_smoothing(img, N=N, sigma=sigma)

    # Display the smoothed image
    # display_image(gaussian_smoothing_img)

    # Compute the image gradients
    gradients = image_gradient(gaussian_smoothing_img)

    # Display the gradient magnitude and orientation
    # display_image(gradients[0], title='Gradient Magnitude')
    # display_image(gradients[1], title='Gradient Orientation')

    # Find the thresholds
    T_low, T_high = find_threshold(gradients[0], percentage_non_edge=percentage_non_edge)
    # print(f"Low Threshold: {T_low}, High Threshold: {T_high}")

    # Apply non-maxima suppression
    gx = gradients[0] * np.cos(gradients[1])
    gy = gradients[0] * np.sin(gradients[1])
    nms = nonmaxima_suppress(gx, gy, gradients[0], method='quantization')

    # Display the non-maxima suppressed image
    # display_image(nms, title='Non-Maxima Suppression')

    # Perform edge linking
    edge_map = edge_linking(nms> T_low, nms > T_high)

    # Display the edge map
    # display_image(edge_map, cmap='gray')

    return edge_map

def edge_cv2(img: np.ndarray,
             N: int = 5,
             sigma: float = 1.0,
             percentage_non_edge: float = 0.75
            ) -> np.ndarray:
    """
    Canny edge detection via OpenCV, with thresholds chosen
    from the gradient-magnitude histogram.

    Args:
      img                  -- input RGB or grayscale image (uint8)
      N                    -- Gaussian kernel size (odd)
      sigma                -- Gaussian std. dev.
      percentage_non_edge  -- fraction of pixels below high threshold

    Returns:
      edges                -- binary edge map (uint8, 0 or 255)
    """
    # 1) Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # 2) Smooth
    blurred = cv2.GaussianBlur(gray, (N, N), sigmaX=sigma)

    # 3) Compute gradient magnitude (for threshold selection)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    # 4) Find T_low, T_high from histogram of mag
    T_low, T_high = find_threshold(mag, percentage_non_edge)

    # 5) OpenCV Canny
    edges = cv2.Canny(blurred,
                      threshold1=T_low,
                      threshold2=T_high,
                      apertureSize=3,
                      L2gradient=True)

    return edges

def experiment_with_edge_detection(img: np.ndarray,
                                   N_list = [],
                                   sigma_list = [],
                                   percentage_non_edge_list = [],
                                   name:str = 'experiment'
                                  ):
    """
    Experiment with different parameters for edge detection.
    
    Args:
        N_list (list): List of kernel sizes to experiment with.
        sigma_list (list): List of sigma values to experiment with.
        percentage_non_edge_list (list): List of percentage non-edge values to experiment with.
    """
    assert len(N_list) == len(sigma_list) == len(percentage_non_edge_list), \
        "All lists must have the same length."
    
    # Detect edges with different parameters
    edge_imgs = []
    for N, sigma, percentage_non_edge in zip(N_list, sigma_list, percentage_non_edge_list):
        edge_imgs.append(edge(img, N=N, sigma=sigma, percentage_non_edge=percentage_non_edge))
    
    # Display the results
    fig, axs = plt.subplots(1, 1+len(N_list), figsize=(4*(1+len(N_list)), 4.5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    for i, (N, sigma, percentage_non_edge) in enumerate(zip(N_list, sigma_list, percentage_non_edge_list)):
        axs[i+1].imshow(edge_imgs[i], cmap='gray')
        axs[i+1].set_title(f'N={N}, sigma={sigma}, p={percentage_non_edge}')

    # Deal with layout
    for ax in axs:
        ax.axis('off')  # Hide axes
    plt.tight_layout()

    # Create a directory to save the results
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'results', name+'.png'), bbox_inches='tight', dpi=300)
    plt.show()
    return None
        
def compare_with_cv2(img: np.ndarray,
                     N: int = 5,
                     sigma: float = 1.0,
                     percentage_non_edge: float = 0.75,
                     name:str = 'compare'
                     ) -> None:
    """
    Compare custom edge detection with OpenCV's Canny.

    Args:
        img (np.ndarray): input image, shape (H, W) or (H, W, C).
        N (int): size of the Gaussian kernel (must be odd).
        sigma (float): standard deviation of the Gaussian.
        percentage_non_edge (float): fraction of pixels to treat as non‐edges (0 < p < 1).
    """
    # Custom edge detection
    custom_edges = edge(img, N=N, sigma=sigma, percentage_non_edge=percentage_non_edge)
    # OpenCV edge detection
    opencv_edges = edge_cv2(img, N=N, sigma=sigma, percentage_non_edge=percentage_non_edge)
    # Display the results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(custom_edges, cmap='gray')
    axs[1].set_title('My Edge Detection')
    axs[2].imshow(opencv_edges, cmap='gray')
    axs[2].set_title('OpenCV Canny Edge Detection')

    # Deal with layout
    for ax in axs:
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    # Create a directory to save the results
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'results', name+'.png'), bbox_inches='tight', dpi=300)
    plt.show()
    return None

def single_edge_detection(img: np.ndarray,
                          N: int = 5,
                          sigma: float = 1.0,
                          percentage_non_edge: float = 0.75,
                          name:str = 'single_edge_detection'
                         ) -> np.ndarray:
    """
    Perform edge detection on a single image.

    Args:
        img (np.ndarray): input image, shape (H, W) or (H, W, C).
        N (int): size of the Gaussian kernel (must be odd).
        sigma (float): standard deviation of the Gaussian.
        percentage_non_edge (float): fraction of pixels to treat as non‐edges (0 < p < 1).

    Returns:
        edges (np.ndarray): binary edge map.
    """
    # Apply Gaussian smoothing
    gaussian_smoothing_img = gaussian_smoothing(img, N=N, sigma=sigma)

    # Display the smoothed image
    # display_image(gaussian_smoothing_img)

    # Compute the image gradients
    gradients = image_gradient(gaussian_smoothing_img)

    # Display the gradient magnitude and orientation
    # display_image(gradients[0], title='Gradient Magnitude')
    # display_image(gradients[1], title='Gradient Orientation')

    # Find the thresholds
    T_low, T_high = find_threshold(gradients[0], percentage_non_edge=percentage_non_edge)
    # print(f"Low Threshold: {T_low}, High Threshold: {T_high}")

    # Apply non-maxima suppression
    gx = gradients[0] * np.cos(gradients[1])
    gy = gradients[0] * np.sin(gradients[1])
    nms = nonmaxima_suppress(gx, gy, gradients[0], method='quantization')

    # Display the non-maxima suppressed image
    # display_image(nms, title='Non-Maxima Suppression')

    # Perform edge linking
    edge_map = edge_linking(nms> T_low, nms > T_high)

    # Display the edge map
    # display_image(edge_map, cmap='gray')

    fig, axs = plt.subplots(1, 5, figsize=(20, 4.5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(gaussian_smoothing_img, cmap='gray')
    axs[1].set_title('Gaussian Smoothing')
    axs[2].imshow(gradients[0], cmap='gray')
    axs[2].set_title('Gradients')
    axs[3].imshow(nms, cmap='gray')
    axs[3].set_title('NMS')
    axs[4].imshow(edge_map, cmap='gray')
    axs[4].set_title('Edge Map')

    # Deal with layout
    for ax in axs:
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    # Create a directory to save the results
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'results', name+'_detection.png'), bbox_inches='tight', dpi=300)
    plt.show()

    return gaussian_smoothing_img, gradients, nms>=T_high, nms>=T_low, edge_map


def main():
    name = 'pointer1'
    image_path = os.path.join(os.path.dirname(__file__), 'instructions', name+'.bmp')
    img = load_image(image_path, img_format='rgb')

    gaus, grad, e_high, e_low, e = single_edge_detection(img,
                                                         N=5,
                                                         sigma=1.0,
                                                         percentage_non_edge=0.75,
                                                         name=name)

    # compare_with_cv2(img,
    #                  N=5,
    #                  sigma=1.0,
    #                  percentage_non_edge=0.75,
    #                  name=name+'_compare')

    # experiment_with_edge_detection(img,
    #                                N_list=[3, 5, 7],
    #                                sigma_list=[1.0, 1.0, 1.0],
    #                                percentage_non_edge_list=[0.75, 0.75, 0.75],
    #                                name=name+'_changing_N')
    # experiment_with_edge_detection(img,
    #                                N_list=[5, 5, 5],
    #                                sigma_list=[0.5, 1.0, 1.5],
    #                                percentage_non_edge_list=[0.75, 0.75, 0.75],
    #                                name=name+'_changing_sigma')
    # experiment_with_edge_detection(img,
    #                                N_list=[5, 5, 5],
    #                                sigma_list=[1.0, 1.0, 1.0],
    #                                percentage_non_edge_list=[0.25, 0.5, 0.75],
    #                                name=name+'_changing_percentage_non_edge')



if __name__ == "__main__":
    main()
