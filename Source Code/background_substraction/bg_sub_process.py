import skimage
import cv2
from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep
from skimage.morphology import disk
from skimage.filters.rank import median
from tqdm import tqdm


def get_processed_images(bg_img_dir_path, image_dir_path, final_datasets):
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
    
    count = 1
    pbar_length = len(os.listdir(image_dir_path))
    pbar = tqdm(total=pbar_length)
    for img_file in os.listdir(image_dir_path):
        #reading the target image for each category
        target_image = skimage.img_as_ubyte(io.imread(os.path.join(image_dir_path, img_file)))
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
        cv2.imwrite(os.path.join(final_datasets, str(count)+'.jpg'), out_eroded)
        pbar.update(1)
        count += 1
    pbar.close()

def check_or_create_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
            print("[success] created directory : %s" % dir_path)
        except Exception as e:
            print("[abort] unable to create directory : %s, error : %s" % (dir_path, str(e)))
            return False
    else:
        print("[info] directory is already present.")
    return True

def initialize_datasets_creation():
    root_path = os.path.abspath("./Dataset/")
    output_datasets = os.path.join(root_path, "binary_images")
    datasets_path = os.path.join(root_path, "Images") 
    bg_dir_name = '_BG'
    bg_dir_path = os.path.join(datasets_path, bg_dir_name)
    for category_dir in os.listdir(datasets_path):
        if category_dir == bg_dir_name:
            #skipping the background images
            continue
        print("working on the category : %s" % (category_dir))
        category_image_dir = os.path.join(datasets_path, category_dir)
        final_datasets = os.path.join(output_datasets, category_dir)
        if check_or_create_directory(final_datasets):
            get_processed_images(bg_dir_path, category_image_dir, final_datasets)

if __name__ == "__main__":
    initialize_datasets_creation()