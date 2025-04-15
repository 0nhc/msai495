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

def _get_neighbour_boundaries(u:int,
                    v:int,
                    shape:np.ndarray.shape):
    # get the boundaries of the neighbors
    ulb = u-1
    uub = u+1
    vlb = v-1
    vub = v+1
    # check if the pixel is on the border
    # if the pixel is on the border, change the lb or ub
    if u == 0:
        ulb = 0
    if u == shape[0]-1:
        uub = shape[0]-1
    if v == 0:
        vlb = 0
    if v == shape[1]-1:
        vub = shape[1]-1
    return ulb, uub, vlb, vub

def _get_label(neighbors:np.ndarray):
    return np.max(neighbors)

def ccl(image_array:np.ndarray):
    """
    Perform connected component labeling on a binary image.
    
    Args:
        image_array (np.ndarray): The binary image array.
    
    Returns:
        np.ndarray: The labeled image array.
        int: The number of connected components found.
    """
    visited_pixels = []
    groups = []

    # go through all the pixels of the image
    for u in range(image_array.shape[0]):
        for v in range(image_array.shape[1]):
            # if the pixel is visited, do nothing
            if (u, v) in visited_pixels:
                continue
            else:
                # if the pixel is a foreground pixel
                if image_array[u, v] == 1:
                    # create a new group
                    groups.append([])
                    # add the pixel to the group
                    groups[-1].append((u, v))
                    # add the pixel to the visited pixels
                    visited_pixels.append((u, v))
                    # neighbours to be visited
                    neighbors_to_visit = []

                    # get the neighbors of the pixel
                    ulb, uub, vlb, vub = _get_neighbour_boundaries(u, v, image_array.shape)
                    neighbors = image_array[ulb:uub+1, vlb:vub+1]
                    neighbors[1, 1] = 0
                    # check if other foreground pixels are in the neighbors
                    if np.any(neighbors == 1):
                        # add the pixels into neighbors_to_visit
                        for i in range(neighbors.shape[0]):
                            for j in range(neighbors.shape[1]):
                                if neighbors[i, j] == 1:
                                    # check if the pixel is already visited
                                    if (ulb+i, vlb+j) not in visited_pixels:
                                        neighbors_to_visit.append((ulb+i, vlb+j))
                    while True:
                        # visit the foreground neighbor recursively
                        visiting = neighbors_to_visit.pop(0)
                        # add the pixel to the group
                        groups[-1].append(visiting)
                        # add the pixel to the visited pixels
                        visited_pixels.append(visiting)

                        # get the neighbors of the pixel
                        ulb, uub, vlb, vub = _get_neighbour_boundaries(visiting[0], visiting[1], image_array.shape)
                        neighbors = image_array[ulb:uub+1, vlb:vub+1]
                        neighbors[1, 1] = 0
                        # check if other foreground pixels are in the neighbors
                        if np.any(neighbors == 1):
                            # add the pixels into neighbors_to_visit
                            for i in range(neighbors.shape[0]):
                                for j in range(neighbors.shape[1]):
                                    if neighbors[i, j] == 1:
                                        # check if the pixel is already visited
                                        if (ulb+i, vlb+j) not in visited_pixels:
                                            # check if the pixel is already going to be visited
                                            if (ulb+i, vlb+j) not in neighbors_to_visit:
                                                neighbors_to_visit.append((ulb+i, vlb+j))
                        # check if there are no more neighbors to visit
                        if len(neighbors_to_visit) == 0:
                            print(f"No more neighbors to visit. Number of groups: {len(groups)}")
                            break
    label = 1
    for group in groups:
        # assign each group a different color in the image
        for pixel in group:
            image_array[pixel[0], pixel[1]] = label
        label += 1
    
    return image_array, len(groups)

def main():
    """
    Main function to load and display an image.
    """
    # Example usage
    path = os.path.join(os.path.dirname(__file__), 'instruction', 'gun.bmp')
    img = load_image(path)
    # display_image(img)
    label_img, num_of_groups = ccl(img)
    print(f"Number of groups: {num_of_groups}")
    display_image(label_img)

if __name__ == "__main__":
    main()
