import skimage
import cv2
from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from math import atan,atan2,pi
import sys

#returns labelled image
def get_fingers_labelled_image(rotated_image,palm_removed,centre_xy):    
    _, labelled = cv2.connectedComponents(rotated_image,connectivity=8)
    pp_color = rotated_image[centre_xy[0]][centre_xy[1]]
    temp = np.copy(labelled)
    temp = temp.astype(int)
    temp[temp==pp_color] = 999
    temp[temp!=999] = 0
    hand_pixels_count = np.count_nonzero(temp)

    num_components, labelled_fingers = cv2.connectedComponents(palm_removed,connectivity=8)
    unique_labels = np.unique(labelled_fingers)
    unique_labels = unique_labels[1:]

    finger_pixels = []
    for label in unique_labels:
        temp = np.copy(labelled_fingers)
        temp = temp.astype(int)
        temp[temp==label] = 999
        temp[temp!=999] = 0  
        finger_pixels_count = np.count_nonzero(temp)
        finger_pixels.append((finger_pixels_count,label))

    for ind in range(len(finger_pixels)):
        count,label = finger_pixels[ind]
        if (count <= hand_pixels_count*0.0020):
            labelled_fingers[labelled_fingers == label] = 0
    
    return labelled_fingers