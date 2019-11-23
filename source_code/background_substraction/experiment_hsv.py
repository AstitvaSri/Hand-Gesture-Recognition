import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
import skimage
from skimage.filters.rank import median
from skimage.morphology import disk
import os


def fill_contours(target_image):
    contour,hier = cv2.findContours(target_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(target_image, [cnt], 0, 255, -1)
    return target_image

def conventional_hand_detection(average_bg_img, image_path):
     #reading the target image for each category
    target_image = skimage.img_as_ubyte(io.imread(os.path.abspath(image_path)))
    #calculating average of image across RGB channels 
    avg_target_result = (average_bg_img[:,:,0] - target_image[:,:,0]) + average_bg_img[:,:,1] - target_image[:,:,1] + average_bg_img[:,:,2] - target_image[:,:,2]
    final_result = avg_target_result/3
    #subtracting the background using threshold values and removing noise using median filter
    binary_img = np.zeros(final_result.shape)
    binary_img[final_result>=17] = 1
    binary_img[final_result<17] = -1
    binary_img = median(binary_img, disk(3))
    #applying the closing on binary image  
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    out_dilated = cv2.dilate(binary_img, se, iterations = 5)
    out_eroded = cv2.erode(out_dilated, se, iterations = 5)
    
    return out_eroded

def hsv_ycrcb_skin_mask(img_path):
    img=cv2.imread(os.path.abspath(img_path))
    
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 55, 0), (200, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, se)

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, se)

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = fill_contours(global_mask)

    return global_mask

#Open a simple image
bg_img_dir_path = os.path.abspath("C:/Users/tusha/Desktop/MS/dip/Hand-Gesture-Recognition/dataset/Images/_BG")
bg_img_collection = [] 
for img_file in os.listdir(bg_img_dir_path):
    bg_img_collection.append(skimage.img_as_ubyte(io.imread(os.path.join(bg_img_dir_path, img_file))))

final_bg_img = np.zeros(bg_img_collection[0].shape)
for i in range(len(bg_img_collection)):
    final_bg_img[:,:,0] = final_bg_img[:,:,0] + bg_img_collection[i][:,:,0]
    final_bg_img[:,:,1] = final_bg_img[:,:,1] + bg_img_collection[i][:,:,1]
    final_bg_img[:,:,2] = final_bg_img[:,:,2] + bg_img_collection[i][:,:,2] 

#calculating the average of all images for background images
average_bg_img = final_bg_img/len(bg_img_collection)


# file_path="./dataset/images/Stop/Stop_24.jpg"
# fig = plt.figure(figsize=(20, 20))
# ax0 = fig.add_subplot(1, 2, 1)
# ax0.set_title("skin color detection")
# mask1 = hsv_ycrcb_skin_mask(file_path)
# ax0.imshow(mask1, cmap='gray')

# ax1 = fig.add_subplot(1, 2, 2)
# ax1.set_title("convention constant bg substraction")
# mask2 = conventional_hand_detection(average_bg_img, file_path)
# ax1.imshow(mask2, cmap='gray')


root_path = os.path.abspath("C:/Users/tusha/Desktop/MS/dip/Hand-Gesture-Recognition/dataset")
datasets_path = os.path.join(root_path, "Images")
bg_dir_name = '_BG'
bg_dir_path = os.path.join(datasets_path, bg_dir_name)

# file_path="C:/Users/tusha/OneDrive/Pictures/Camera Roll/1.jpg"
# fig = plt.figure(figsize=(20, 20))
# ax0 = fig.add_subplot(1, 2, 1)
# ax0.set_title("skin color detection")
# mask1 = hsv_ycrcb_skin_mask(file_path)
# ax0.imshow(mask1, cmap='gray')
# plt.show()

# ax1 = fig.add_subplot(1, 2, 2)
# ax1.set_title("convention constant bg substraction")
# mask2 = conventional_hand_detection(average_bg_img, file_path)
# ax1.imshow(mask2, cmap='gray')

threshold = 10
for category_dir in os.listdir(datasets_path):
    if category_dir == bg_dir_name:
        #skipping the background images
        continue
    print("working on the category : %s" % (category_dir))
    category_image_dir = os.path.join(datasets_path, category_dir)
    files_names = os.listdir(category_image_dir)
    np.random.shuffle(files_names)
    for index in range(threshold):
        img_file = files_names[index]
        target_image_path = os.path.join(category_image_dir, img_file)
        fig = plt.figure(figsize=(20, 20))
        ax0 = fig.add_subplot(1, 2, 1)
        ax0.set_title("skin color detection for category : %s" % category_dir)
        mask1 = hsv_ycrcb_skin_mask(target_image_path)
        ax0.imshow(mask1, cmap='gray')

        ax1 = fig.add_subplot(1, 2, 2)
        ax1.set_title("convention constant bg substraction for category : %s" % category_dir)
        mask2 = conventional_hand_detection(average_bg_img, target_image_path)
        ax1.imshow(mask2, cmap='gray')
        plt.show()