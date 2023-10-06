from typing import Tuple

import numpy as np


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    
    border = 10
    percent_height = 0.4
    percent_width = 0.5

    height = shape[0]
    partial_height = int(percent_height*height)
    width = shape[1]
    partial_width = int(percent_width*width)

    res[partial_height:height,border:partial_width] = 1
    return res


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")
    
    border = 10
    percent_height = 0.4
    percent_width = 0.5

    height = shape[0]
    partial_height = int(percent_height*height)
    width = shape[1]
    partial_width = int(percent_width*width)

    res[partial_height:height,partial_width:width-border] = 1
    return res
