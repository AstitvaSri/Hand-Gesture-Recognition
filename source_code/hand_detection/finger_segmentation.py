import skimage
import cv2
from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from math import atan,atan2,pi,isnan
import sys


def get_segmentation_points(binimg,centre_xy,wrist_mid):
    binimg[binimg!=0] = 255
    binimg = binimg.astype('uint8')

    cnt,_ = cv2.findContours(binimg,1,2)
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)    
    mid1 = ( (box[0][0]+box[1][0])//2 , (box[0][1]+box[1][1])//2 )
    mid2 = ( (box[1][0]+box[2][0])//2 , (box[1][1]+box[2][1])//2 )
    mid3 = ( (box[2][0]+box[3][0])//2 , (box[2][1]+box[3][1])//2 )
    mid4 = ( (box[3][0]+box[0][0])//2 , (box[3][1]+box[0][1])//2 )

    #determining base mid points
    middle_points = [mid1,mid2,mid3,mid4]
    min_dist_palm_point = max(binimg.shape[0],binimg.shape[1])
    middle_base_point = 0
    for mp in middle_points:
        x,y = mp
        binimg[y][x] = 200
        dist = np.sqrt((centre_xy[0]-x)**2 + (centre_xy[1]-y)**2)
        if dist < min_dist_palm_point:
            min_dist_palm_point = dist
            middle_base_point = mp

    slope = (centre_xy[0]-middle_base_point[0]) / (centre_xy[1]-middle_base_point[0])
    angle = atan(slope)*(180/pi)
    if angle<0:
        angle = 180 + angle
    print(angle)

    binimg[centre_xy[0]][centre_xy[1]] = 255
    plt.imshow(binimg,cmap="gray")
    plt.show()
    return middle_base_point


#returns labelled imagenum_components
def get_fingers_labelled_image(rotated_image,palm_removed,centre_xy,wrist_points):
    wrist_mid = [ (wrist_points[0][0]+wrist_points[1][0])//2 , (wrist_points[0][1]+wrist_points[1][1])//2 ]
    _, labelled = cv2.connectedComponents(rotated_image,connectivity=8)
    pp_color = rotated_image[centre_xy[0]][centre_xy[1]]
    temp = np.copy(labelled)
    temp = temp.astype(int)
    temp[temp==pp_color] = 999
    temp[temp!=999] = 0
    hand_pixels_count = np.count_nonzero(temp)

    _, labelled_fingers = cv2.connectedComponents(palm_removed,connectivity=8)
    unique_labels = np.unique(labelled_fingers)
    unique_labels = unique_labels[1:]

    finger_pixels = []
    fingers_base_mid_points = []
    for label in unique_labels:
        temp = np.copy(labelled_fingers)
        temp = temp.astype(int)
        temp[temp==label] = 999
        temp[temp!=999] = 0         
        fingers_base_mid_points.append(get_segmentation_points(temp,centre_xy,wrist_mid)) 
        finger_pixels_count = np.count_nonzero(temp)
        finger_pixels.append((finger_pixels_count,label))

    for ind in range(len(finger_pixels)):
        count,label = finger_pixels[ind]
        if (count <= hand_pixels_count*0.0020):
            labelled_fingers[labelled_fingers == label] = 0

    return labelled_fingers