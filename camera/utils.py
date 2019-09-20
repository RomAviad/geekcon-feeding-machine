import numpy as np


def calculate_naive_mask_bounding_box(img):
    """
    Takes an image of an isolated object on black background
    and returns its bounding box corners
    :param img:
    :return:
    """
    min_x, min_y, max_x, max_y = 0, 0, 0, 0
    non_zeros = np.nonzero(img)
    if np.any(non_zeros):
        min_x = non_zeros[0].min()
        max_x = non_zeros[0].max()
        min_y = non_zeros[1].min()
        max_y = non_zeros[1].max()

    return min_x, min_y, max_x, max_y