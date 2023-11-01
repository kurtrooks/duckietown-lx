from typing import Tuple

import numpy as np
import cv2

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """


    steer_matrix_left = np.ones((shape[0],shape[1]))

    for y in range(0,shape[1]):
        steer_matrix_left[:,y] =  (y/shape[1])

    return -steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    steer_matrix_right = np.zeros((shape[0],shape[1]))
    
    for y in range(0,shape[1]):
        steer_matrix_right[:,y] = 1- y/shape[1]

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    height, width, _ = image.shape
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    sigma = 3
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # Sobel (edge detect)
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)
 
    """ Masks """
    # Gradient
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    # Ground mask 
    mask_ground = np.ones(img.shape, dtype=np.uint8)
    mask_ground[:int(width/2)-50,:] = 0

    # Left / right mask
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0
  
    # Mag Threshold
    threshold = 50
    mask_mag = (Gmag > threshold)

    # Color Mask
    white_lower_hsv = np.array([0,0,100])  
    white_upper_hsv = np.array([179,60,255]) 
    yellow_lower_hsv = np.array([18, 96, 115])
    yellow_upper_hsv = np.array([40,255,255])

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
   
    # Output
    mask_left_edge= mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg *mask_yellow #* mask_ground
    mask_right_edge= mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg *mask_white #* mask_ground

    return mask_left_edge, mask_right_edge
