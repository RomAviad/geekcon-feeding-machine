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


def get_largest_face_polygon(faces):
    largest_area = 0
    left, right, top, bottom = 0, 0, 0, 0
    for face in faces:
        f_left = face[0]
        f_top = face[1]
        f_right = face[0] + face[2]
        f_bottom = face[1] + face[3]
        area = abs(f_right - f_left) * abs(f_top - f_bottom)
        if area > largest_area:
            largest_area = area
            left = f_left
            right = f_right
            top = f_top
            bottom = f_bottom
    return left, top, right, bottom