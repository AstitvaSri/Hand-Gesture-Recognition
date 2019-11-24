import skimage
import cv2
from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as LA
from datetime import datetime
from math import atan,atan2,pi
import sys
import math
#adding root project to system path
root_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_project)
from source_code.hand_detection.palm_detection import extract_palm_mask
from skimage.draw import circle_perimeter, line


def get_angle(point1, point2):
    e = 0
    #avoid divide by zero
    if point2[0] - point1[0] == 0:
        e = 1e-6
    angle = np.rad2deg(np.arctan2(point2[1] - point1[1], point2[0] - point1[0] + e))
    return angle

def get_finger_segmentation_points(input_binary_image, palm_point, wrist_point_one, wrist_point_two):
    input_binary_image[input_binary_image!=0] = 255
    input_binary_image = input_binary_image.astype('uint8')

    cnt, _ = cv2.findContours(input_binary_image, 1, 2)
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #converting to row and column convention
    box[:, [0, 1]] = box[:, [1, 0]]

    #finding midpoints at each edge in rectangle    
    mid1 = ( (box[0][0]+box[1][0])//2 , (box[0][1]+box[1][1])//2 )
    mid2 = ( (box[1][0]+box[2][0])//2 , (box[1][1]+box[2][1])//2 )
    mid3 = ( (box[2][0]+box[3][0])//2 , (box[2][1]+box[3][1])//2 )
    mid4 = ( (box[3][0]+box[0][0])//2 , (box[3][1]+box[0][1])//2 )

    #determining base mid points
    middle_points = [mid1 , mid2, mid3, mid4]
    min_dist_palm_point = max(input_binary_image.shape[0], input_binary_image.shape[1])
    max_dist_palm_point = 0
    middle_base_point = None
    middle_top_point = None
    for mp in middle_points:
        x, y = mp
        input_binary_image[x][y] = 200
        dist = LA.norm((np.array(palm_point) - np.array([x, y])), 2)
        if dist < min_dist_palm_point:
            min_dist_palm_point = dist
            middle_base_point = mp
        if dist > max_dist_palm_point:
            max_dist_palm_point = dist
            middle_top_point = mp

    #from palm_point to centre of boxing figures
    theta_1 = get_angle(middle_base_point, palm_point)
    theta_2 = get_angle(wrist_point_one, wrist_point_two)
    
    #calculates the angle between points 
    theta = theta_1 - theta_2
    return middle_base_point, middle_top_point, theta, box

#returns labelled image
def get_fingers_labelled_image(rotated_image, palm_removed, centre_xy):    
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

def detect_thumb(detected_fingers, debug=False):
    thumb_label = -1
    right_aligned = 0
    threshold1, threshold2 = 15, 171
    #if five components are detected, find the lowest angle
    if len(detected_fingers) == 5:
        done=False
        least_angle = 180
        least_angle_label = thumb_label
        least_right_aligned = 0
        for label, data_dict in detected_fingers.items():
            theta = data_dict['angle']
            minor_angle = theta
            if minor_angle > 90:
                minor_angle = 180 - minor_angle
            if minor_angle < least_angle:
                least_angle = minor_angle
                least_angle_label = label
                least_right_aligned = 1 if theta < 90 else -1
            if not done and (theta < threshold1 or theta > threshold2):
                done = True
                right_aligned = 1 if theta < threshold1 else -1
                thumb_label = label
        #if no thumb is detected use the least angle finger as thumb
        if thumb_label == -1:
            if debug:
                print("[DEBUG] detect_thumb : no finger met threshold value in five finger, using the least angle")
            right_aligned = least_right_aligned
            thumb_label = least_angle_label
    else:
        for label, data_dict in detected_fingers.items():
            theta = data_dict['angle']
            if theta < threshold1 or theta > threshold2:
                thumb_label = label
                right_aligned = 1 if theta < threshold1 else -1
                break
    
    return thumb_label!=-1, thumb_label, right_aligned

def extract_palm_line(input_image, wrist_point_one, wrist_point_two, segmented_image, thumb_label, debug=False):
    start_time = datetime.utcnow()
    x_max, y_max = input_image.shape
    x_min = min(int(wrist_point_one[0]), int(wrist_point_two[1]), x_max)
    y_min = 0
    black_area_threshold, white_area_threshold = 2, 2
    #starting from the bottom removing 10 pixels
    for i in range(x_min-10, -1, -1):
        components = 0
        start_point, end_point = 0, y_max
        black_pixel_area = []
        white_pixel_area = []
        last_change = 0
        flag = False
        for j in range(0, y_max):
            prev_pixel_value = 0
            if j != 0:
                prev_pixel_value = input_image[i, j-1]
            if prev_pixel_value == input_image[i, j]:
                if segmented_image[i, j] == thumb_label and components<=2:
                    flag = True
                continue
            if prev_pixel_value == 0 and input_image[i, j] == 255:
                if segmented_image[i, j] != thumb_label:
                    black_pixel_area.append(j-last_change)
                    components+=1
                if start_point == 0 and segmented_image[i, j] != thumb_label:
                    start_point = j
                last_change = j
            if prev_pixel_value == 255 and input_image[i, j] == 0:
                if segmented_image[i, j] != thumb_label:
                    white_pixel_area.append(j - last_change)
                    end_point = j
                last_change = j
            #update the previous pixels
            prev_pixel_value = input_image[i, j]
        if components >= 2:
            #check black pixel area
            for area in black_pixel_area:
                if area < black_area_threshold:
                    flag = True
                    break
            #check white pixel area
            for area in white_pixel_area:
                if area < white_area_threshold:
                    flag = True
                    break
            if flag:
                continue
            thumb_point1 = [i, start_point]
            thumb_point2 = [i, end_point]
            if debug:
                time_elapsed = (datetime.utcnow() - start_time).total_seconds()
                print("[DEBUG] extract_palm_points : finished in %.2f secs" % (time_elapsed))
            return [thumb_point1, thumb_point2]
    # this won't happen written to avoid runtime errors
    if debug:
        time_elapsed = (datetime.utcnow() - start_time).total_seconds()
        print("[DEBUG] extract_palm_points : finished in %.2f secs" % (time_elapsed))
    return [None, None]   

# works on the rotated image and palm mask
def detect_finger(input_image, palm_mask, palm_point, wrist_point_one, wrist_point_two, hand_area, debug=False, visual_debug=False):
    raw_finger_segment = (input_image - palm_mask)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(raw_finger_segment, connectivity=4)
    intermediate_image = np.zeros(input_image.shape)
    previously_labeled = -1
    #label zero is assigned to background of the image
    detected_fingers = {}
    for component_count in range(1, nb_components):
        ratio = stats[component_count][-1]/float(hand_area)
        if debug:
            print("[DEBUG] detect_finger : label - %d, area_ratio - %f, centre coordinates - %s" % (component_count, ratio, centroids[component_count]))
        # detected a finger
        if 0.025 <= ratio and ratio <= 0.2:
            #creating an empty image
            if previously_labeled > 0:
                intermediate_image[output == previously_labeled] = 0
            #update the image
            intermediate_image[output == component_count] = 255
            #update the label
            previously_labeled = component_count
            #calculate finger information
            middle_base_point, middle_top_point, theta, box = get_finger_segmentation_points(intermediate_image, palm_point, wrist_point_one, wrist_point_two)
            detected_fingers[component_count] = {
                "centre_coordinates" : centroids[component_count],
                "area" : stats[component_count][-1],
                "middle_base_point" : middle_base_point,
                "middle_top_point" : middle_top_point,
                "angle" : theta,
                "rectangular boundary" : box,
                "is_thumb" : False,
            }
            if debug:
                print("[DEBUG] detect_finger : --> adding %d to fingers region" % component_count)
    
    is_thumb_present, thumb_label, right_hand = detect_thumb(detected_fingers, debug=True)
    if is_thumb_present:
        if debug:
            print("[DEBUG] thumb detected with label : %d" % thumb_label)
        detected_fingers[thumb_label].update({'is_thumb' : True, 'hand_orientation' : right_hand})
    #single fingure doesn't require the palm point
    palm_line_point1, palm_line_point2 = None, None
    if len(detected_fingers) > 1:
        palm_line_point1, palm_line_point2 = extract_palm_line(input_image, wrist_point_one, wrist_point_two, output, thumb_label, debug=True)
    if visual_debug:
        segmented_image = np.zeros(output.shape)
        for label in detected_fingers:
            segmented_image[output == label] = label
        fig = plt.figure()
        ax0 = fig.add_subplot(2, 2, 1)
        one_title = "original image"
        if palm_line_point1 is not None and palm_line_point2 is not None:
            one_title = "original image with palm line"
        ax0.set_title(one_title)
        if palm_line_point1 is not None and palm_line_point2 is not None:
            display_image = np.stack([input_image, input_image, input_image], axis=2)
            cv2.line(display_image, (palm_line_point1[1], palm_line_point1[0]), (palm_line_point2[1], palm_line_point2[0]), (255, 0, 0), 5)
            ax0.imshow(display_image)
        else:
            ax0.imshow(input_image, cmap='gray')
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.set_title("after palm segmentation")
        ax1.imshow(raw_finger_segment, cmap='gray')
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.set_title("%d components" % (len(detected_fingers)))
        ax2.imshow(segmented_image)
        #preparing for the final display
        final_display = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8)
        #palm point
        cv2.circle(final_display, (int(palm_point[1]), int(palm_point[0])), 13, (0, 0, 255), 3)
        #plot lines
        if palm_line_point1 is not None and palm_line_point2 is not None:
            cv2.line(final_display, (palm_line_point1[1], palm_line_point1[0]), (palm_line_point2[1], palm_line_point2[0]), (128, 0, 128), 3)
        for label, data_dict in detected_fingers.items():
            middle_base_point = data_dict['middle_base_point']
            centre_coordinates = data_dict['centre_coordinates']
            middle_top_point = data_dict['middle_top_point']
            box = data_dict["rectangular boundary"]
            box[:, [0, 1]] = box[:, [1, 0]]
            cv2.line(final_display, (int(palm_point[1]), int(palm_point[0])), (int(middle_base_point[1]), int(middle_base_point[0])), (90, 255, 0), 2)
            cv2.circle(final_display, (int(middle_base_point[1]), int(middle_base_point[0])), 2, (255, 255, 0), 2)
            cv2.line(final_display, (int(centre_coordinates[0]), int(centre_coordinates[1])), (int(middle_base_point[1]), int(middle_base_point[0])), (90, 255, 0), 2)
            cv2.circle(final_display, (int(middle_top_point[1]), int(middle_top_point[0])), 2, (255, 255, 0), 2)
            cv2.drawContours(final_display, [box], 0, (255, 0, 0), 2)
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.set_title("%d fingers with %s thumb" % (len(detected_fingers) - (1 if is_thumb_present else 0), "one" if is_thumb_present else "no"))
        ax3.imshow(final_display)
        plt.show()
    hand_orientation = "can't say"
    if right_hand == 1:
        hand_orientation = "right"
    if right_hand == -1:
        hand_orientation = "left"
    print("[INFO] detect_finger : %d finger detected with %s thumb. probable hand orientation : %s" % (len(detected_fingers) - (1 if is_thumb_present else 0), "one" if is_thumb_present else "no", hand_orientation))
    print('--'*32)

img = img_as_ubyte(io.imread(os.path.abspath('C:/Users/tusha/Desktop/MS/dip/Hand-Gesture-Recognition/dataset/new_binary_images/5/50.jpg')))
palm_mask, rotated_img, rotated_palm_points, rotated_wrist_point_one, rotated_wrist_point_two, hand_area = extract_palm_mask(img, debug=False, visual_debug=True)
detect_finger(rotated_img, palm_mask, rotated_palm_points, rotated_wrist_point_one, rotated_wrist_point_two, hand_area, visual_debug=True)

# threshold = 5
# root_dir = os.path.abspath('C:/Users/tusha/Desktop/MS/dip/Hand-Gesture-Recognition/dataset/new_binary_images/')
# for dir_name in os.listdir(root_dir):
#     if dir_name == '_BG' or dir_name != '5':
#         continue
#     print("working on the dataset %s" % (dir_name))
#     file_names = os.listdir(os.path.join(root_dir, dir_name))
#     np.random.shuffle(file_names)
#     for img_file_name in file_names[:threshold]:
#         img_file_path = os.path.join(root_dir, dir_name, img_file_name)
#         img = img_as_ubyte(io.imread(os.path.abspath(img_file_path)))
#         palm_mask, rotated_img, rotated_palm_points, rotated_wrist_point_one, rotated_wrist_point_two, hand_area = extract_palm_mask(img, debug=False, visual_debug=True)
#         detect_finger(rotated_img, palm_mask, rotated_palm_points, rotated_wrist_point_one, rotated_wrist_point_two, hand_area, visual_debug=True)