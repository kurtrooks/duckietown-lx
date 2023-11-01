#!/usr/bin/env python3

from visual_servoing_activity import *

import numpy as np
import cv2
from matplotlib import pyplot as plot

img_left = get_steer_matrix_left_lane_markings((480,512))
img_right = get_steer_matrix_right_lane_markings((480,512))

#mask1 = np.repeat(np.tile(np.linspace(0,1, img1.shape[1]), (img1.shape[0], 1))[:, :, np.newaxis], 3, axis=2)

fig = plot.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img_left)

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img_right)

plot.show()



