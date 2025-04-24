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
    img = np.array(img)
    return img

def display_image(image_array:np.ndarray,
                  cmap:str = 'gray',
                  title:str = None): 
    """
    Display an image using matplotlib.
    
    Args:
        image_array (np.ndarray): The image array to display.
    """
    plt.imshow(image_array, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')  # Hide axes
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.show()

def histogram_equalize(img: np.ndarray):
    """
    Perform global histogram equalization on a grayscale image.
    """
    # 1) Compute histogram (counts of each intensity 0–255)
    hist = []
    for _ in range(256):
        hist.append(0)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1

    # 2) Normalize to get the Probability Distribution (PD)
    N = h * w
    # pdf = [count / N for count in hist]
    for idx, count in enumerate(hist):
        # Normalize the histogram to get the PD
        hist[idx] = count / N
    pd = hist

    # 3) Compute the Cumulative Distribution (CD)
    cd = []
    for _ in range(256):
        cd.append(0)
    cumulative = 0.0
    for i in range(256):
        cumulative += float(pd[i])
        cd[i] = cumulative

    # 4) Build the mapping: new_intensity = round(255 * CD(old_intensity))
    # mapping = [round(c * 255) for c in cd]
    mapping = []
    for c in cd:
        # Scale the cumulative distribution to the range [0, 255]
        c = min(round(c * 255), 255)
        c = max(c, 0)
        mapping.append(c)

    # 5) Apply the mapping to create the equalized image
    he_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            he_img[i, j] = mapping[img[i, j]]
    
    # 6) Get the Probability Distribution (PD) of the equalized image
    he_hist = []
    for _ in range(256):
        he_hist.append(0)
    h, w = he_img.shape
    for i in range(h):
        for j in range(w):
            he_hist[he_img[i, j]] += 1
    # Normalize the histogram to get the PD
    for idx, count in enumerate(he_hist):
        # Normalize the histogram to get the PD
        he_hist[idx] = count / (h * w)
    pd_he = he_hist

    return he_img, pd, cd, pd_he

def plot_histogram(hist: list,
                   title: str = 'Histogram',
                   xlabel: str = 'Intensity',
                   ylabel: str = 'Frequency'):
    """
    Plot the histogram of an image.
    
    Args:
        hist (list): The histogram data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.bar(range(256), hist, width=1, color='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 255)
    plt.grid()
    plt.show()

def plot_continuous(x: np.ndarray,
                    y: np.ndarray,
                    title: str = 'Plot',
                    xlabel: str = 'X-axis',
                    ylabel: str = 'Y-axis'):
    """
    Plot a continuous function.
    
    Args:
        x (np.ndarray): The x values.
        y (np.ndarray): The y values.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.plot(x, y, color='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def lighting_correction(img: np.ndarray,
                        method:str = 'linear'):
    """
    Perform lighting correction on the image.

    Args:
        img (np.ndarray): The input image.
        method (str): The method for lighting correction ('linear' or 'quadratic').

    Returns:
        np.ndarray: The corrected image.
    """
    h, w = img.shape
    # x goes from 0→1 horizontally, y goes from 0→1 vertically
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    # make 2D grids
    X, Y = np.meshgrid(xs, ys)
    illum = np.zeros((h, w), dtype=np.float64)
    if method == 'linear':
        # build design matrix [x y 1]
        A_lin = np.stack([X.ravel(), Y.ravel(), np.ones(h*w)], axis=1)
        b = img.ravel().astype(np.float64)
        # solve for [a,b,c]
        coeffs_lin, *_ = np.linalg.lstsq(A_lin, b, rcond=None)
        a, b, c = coeffs_lin
        illum = (a*X + b*Y + c)
    elif method == 'quadratic':
        # design: [x^2, y^2, x*y, x, y, 1]
        A_quad = np.stack([
            X.ravel()**2,
            Y.ravel()**2,
            (X*Y).ravel(),
            X.ravel(),
            Y.ravel(),
            np.ones(h*w),
        ], axis=1)
        b = img.ravel().astype(np.float64)
        coeffs_q, *_ = np.linalg.lstsq(A_quad, b, rcond=None)
        a2, b2, c2, d2, e2, f2 = coeffs_q
        illum = (
            a2*X**2 + b2*Y**2 + c2*X*Y +
            d2*X   + e2*Y   + f2
        )
    else:
        raise ValueError("Invalid method. Choose 'linear' or 'quadratic'.")
    
    mean_illum = illum.mean()
    corr = img.astype(np.float64) * (mean_illum / illum)
    corr = np.clip(corr, 0, 255).astype(np.uint8)
    return corr

def main():
    """
    Main function.
    """
    # Load the image
    path = os.path.join(os.path.dirname(__file__), 'instructions', 'moon.bmp')
    img = load_image(path)
    display_image(img)

    # Histogram equalization
    he_img, pd, cd, pd_he  = histogram_equalize(img)
    display_image(he_img)
    plot_histogram(pd, title='Original Histogram', xlabel='Intensity', ylabel='Probability')
    plot_histogram(pd_he, title='Equalized Histogram', xlabel='Intensity', ylabel='Probability')
    plot_continuous(np.arange(256), cd, title='Cumulative Distribution', xlabel='Intensity', ylabel='Cumulative Probability')

    # Lighting correction
    linear_corr = lighting_correction(he_img, method='linear')
    display_image(linear_corr)
    quadratic_corr = lighting_correction(he_img, method='quadratic')
    display_image(quadratic_corr)

if __name__ == "__main__":
    main()
