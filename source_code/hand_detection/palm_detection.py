import skimage
import cv2
from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from math import atan2
import sys
#adding root project to system path
root_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_project)
from source_code.utilities.circle_creation import bresenhamCircle
from source_code.utilities.graph_utils import bfs
import scipy
from skimage.morphology import convex_hull_image
from skimage.draw import circle_perimeter, line


def detect_palm_circle(input_img, debug=False, visual_debug=False):
    start_time = datetime.utcnow()
    #dilation for carrying out correction
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.dilate(input_img, SE, iterations = 1)
    #finding the palm point
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    r,c = dist.shape
    palm_point = dist//np.max(dist)  
    palm_point_location = np.argmax(palm_point)
    palm_x = palm_point_location//c
    palm_y = palm_point_location % c
    if debug:
        print("[DEBUG] detect_palm_circle : found the palm point at location [%d, %d]" % (palm_x, palm_y))
    #finding the inner palm circle
    inner_circle_radius = 1
    inner_circle_status = False
    while inner_circle_radius < min(input_img.shape[0], input_img.shape[1]):
        circle_points = bresenhamCircle(palm_x, palm_y, inner_circle_radius)
        for pixel_points in circle_points:
            #skipping the pixels out of the image boundary
            if pixel_points[0] < 0 or pixel_points[0] >= input_img.shape[0]:
                continue
            if pixel_points[1] < 0 or pixel_points[1] >= input_img.shape[1]:
                continue
            if input_img[pixel_points[0], pixel_points[1]] == 0:
                inner_circle_status = True
        if inner_circle_status:
            break
        inner_circle_radius+=1
    outer_circle_radius = int(1.5 * inner_circle_radius)
    if debug:
        print("[DEBUG] detect_palm_circle : inner circle radius - %d and outer circle radius - %d" % (inner_circle_radius, outer_circle_radius))
    end_time = (datetime.utcnow() - start_time).total_seconds()
    if debug:
        print("[DEBUG] detect_palm_circle : palm circle detection done in %.2f secs." % (end_time))
    
    if visual_debug:
        display_image = np.copy(input_img)
        fig = plt.figure()
        ax0 = fig.add_subplot(2, 2, 1)
        ax0.set_title("original input image")
        ax0.imshow(input_img, cmap='gray')
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.set_title("distance transform")
        ax1.imshow(dist, cmap='gray')
        bresenhamCircle(palm_x, palm_y, outer_circle_radius, color=127, image=display_image)
        display_image[palm_x, palm_y] = 255
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.set_title("palm circle with centre : [%d, %d]" % (palm_x, palm_y))
        ax2.imshow(display_image, cmap='gray')
        plt.show()
    return [palm_x, palm_y], inner_circle_radius, outer_circle_radius
        

def get_boundary_points(sample_points, input_image, debug=False):
    boundary_points = []
    start_time = datetime.utcnow()
    for count, sample in enumerate(sample_points):
        result = bfs(input_image, sample[0], sample[1], 10)
        if debug and False:
            print("[DEBUG] get_boundary_points : index [%d] got boundary point at %s for sample point %s" % (count, result, sample))
        #discard the empty boundary from the bfs output
        if result != [-1, -1]:
            boundary_points.append(result)
    end_time = (datetime.utcnow() - start_time).total_seconds()
    if debug:
        print("[DEBUG] get_boundary_points : boundary points detected in %.2f secs" % (end_time))
    return boundary_points

def wrist_detection(boundary_points):
    max_distance = 0.0
    first_coord_index, second_coord_index = -1, -1
    for i in range(0, len(boundary_points)):
        if i == len(boundary_points) - 1:
            j = 0
        else:
            j = i + 1
        distance = np.linalg.norm(np.array(boundary_points[i] - np.array(boundary_points[j])))
        if distance > max_distance:
            max_distance = distance
            first_coord_index, second_coord_index = i, j
    return boundary_points[first_coord_index], boundary_points[second_coord_index]

# def detect_angle_of_rotation(wrist_points, )

def valid_circle_points(rr, cc, x_max, y_max):
    result_x, result_y = [], []
    for index in range(len(rr)):
        if 0 <= rr[index] < x_max and 0 <= cc[index] < y_max:
            result_x.append(rr[index])
            result_y.append(cc[index])
    return result_x, result_y

def extract_palm_mask(input_image, sample_spacing=20, debug=False, visual_debug=False, vd_last_step=True):
    centre_xy, inner_circle_radius, outer_circle_radius = detect_palm_circle(input_image, visual_debug=visual_debug and not vd_last_step, debug=debug)
    outer_circle_points = bresenhamCircle(centre_xy[0], centre_xy[1], outer_circle_radius)
    #sorting points in clockwise direction
    outer_circle_points = sorted(outer_circle_points, key = lambda c:atan2(c[0] - centre_xy[0], c[1] - centre_xy[1]))
    spacing = max(int((sample_spacing*len(outer_circle_points))/float(360)), 1)
    #subset the sampling space
    sample_points = []
    for i in range(0, len(outer_circle_points), spacing):
        sample_points.append([outer_circle_points[i][0], outer_circle_points[i][1]])
    if debug:
        print("[DEBUG] extract_palm_mask : outer circle point - %d, sample space - %d and sample points - %d" % (len(outer_circle_points), spacing, len(sample_points)))
    boundary_points = get_boundary_points(sample_points, input_image, debug=debug)
    convex_hull, palm_mask = np.zeros(input_image.shape, dtype=np.uint8), np.zeros(input_image.shape, dtype=np.uint8)
    for points in boundary_points:
        convex_hull[points[0], points[1]] = 255
    convex_hull = convex_hull_image(convex_hull)
    palm_mask[convex_hull==True] = 255
    palm_mask[convex_hull==False] = 0
    wrist_point_one, wrist_point_two = wrist_detection(boundary_points)
    palm_mask = scipy.bitwise_and(input_image, palm_mask)
    if visual_debug:
        display_image = np.stack([input_image, input_image, input_image], axis=2)
        # displaying the palm centre
        rr, cc = circle_perimeter(centre_xy[0], centre_xy[1], 3)
        rr, cc = valid_circle_points(rr, cc, display_image.shape[0], display_image.shape[1])
        display_image[rr, cc] = [0, 0, 255]
        # displaying the palm inner circle
        rr, cc = circle_perimeter(centre_xy[0], centre_xy[1], inner_circle_radius)
        rr, cc = valid_circle_points(rr, cc, display_image.shape[0], display_image.shape[1])
        display_image[rr, cc] = [0, 255, 0]
        # displaying the palm outer circle
        rr, cc = circle_perimeter(centre_xy[0], centre_xy[1], outer_circle_radius)
        rr, cc = valid_circle_points(rr, cc, display_image.shape[0], display_image.shape[1])
        display_image[rr, cc] = [0, 255, 128]
        # displaying the wirst line
        rr, cc = line(wrist_point_one[0], wrist_point_one[1], wrist_point_two[0], wrist_point_two[1])
        display_image[rr, cc] = [255, 0, 0]
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.set_title("original input image")
        ax0.imshow(display_image)
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.set_title("palm mask with wrist line")
        ax1.imshow(palm_mask, cmap='gray')
        plt.show()
    return palm_mask

img = img_as_ubyte(io.imread(os.path.abspath('C:/Users/tusha/Desktop/MS/dip/Hand-Gesture-Recognition/dataset/new_binary_images/5/89.jpg')))
palm_mask = extract_palm_mask(img, debug=True, visual_debug=True, vd_last_step=False)
plt.imshow((img - palm_mask), cmap='gray')
plt.show()
