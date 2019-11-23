import skimage
import cv2
from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep
from skimage.morphology import disk
from skimage.filters.rank import median
import math

dir_path = os.path.abspath("../dataset/Images/_BG/")

bg = [] 
for the_file in os.listdir(dir_path):
    bg.append(skimage.img_as_ubyte(io.imread(os.path.join(dir_path, the_file))))

# bg = np.array(bg)
result = np.zeros(bg[0].shape)
for i in range(len(bg)):
    result[:,:,0] = result[:,:,0] + bg[i][:,:,0]
    result[:,:,1] = result[:,:,1] + bg[i][:,:,1]
    result[:,:,2] = result[:,:,2] + bg[i][:,:,2] 

bg_frame = result/len(bg)
# fig = plt.figure(figsize=(40, 30))
dir_path = "../dataset/Images/5/"

batch_size = 20

for the_file in os.listdir(dir_path):
    target_image = skimage.img_as_ubyte(io.imread(os.path.join(dir_path, the_file)))
    avg = (bg_frame[:,:,0] - target_image[:,:,0]) + bg_frame[:,:,1] - target_image[:,:,1] + bg_frame[:,:,2] - target_image[:,:,2]
    avg = avg/3 

    temp = np.zeros(avg.shape)
    temp[avg>=17] = 1
#     temp[avg<17] = -1
    out = median(temp, disk(3))
    
    
    #performing closing operation
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    out_dilated = cv2.dilate(out,SE,iterations = 1)    
    out_eroded = cv2.erode(out_dilated,SE,iterations = 1)
    out = out_eroded
    
    #distance transform    
    dist = cv2.distanceTransform(out, cv2.DIST_L2, 5) 
              
    
    #palm point
    r,c = dist.shape
    palm_point = dist//np.max(dist)  
#     print(np.unique(palm_point))
    palm_point_location = np.argmax(palm_point)
    palm_x = palm_point_location//c
    palm_y = palm_point_location%c
#     print(palm_x,palm_y,palm_point[palm_x][palm_y])
    
    
    #dilating for correction
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    out_dilated = cv2.dilate(out,SE,iterations = 5)
    
    #determining inner circle radius
    a = palm_x
    b = palm_y
    naive_radius = 1    
    found_local_shortest = False
    
    up = [a-1,b]
    down = [a+1,b]
    left = [a,b-1]
    right = [a,b+1]
    diag_up_left = [a-1,b-1]
    diag_up_right = [a-1,b+1]
    diag_down_left = [a+1,b-1]
    diag_down_right = [a+1,b+1]
    
    while(True):
        black_count=0
        thresh = 0
        loc_list = [up,down,left, right, diag_up_left, diag_up_right, diag_down_left, diag_down_right]
        for loc in loc_list:
            if loc[0]>=256 or loc[1]>=c or out_dilated[loc[0]][loc[1]] == 0 :   
                black_count+=1
                found_local_shortest = True
        
        if found_local_shortest == True or palm_x+naive_radius>=r or palm_y+naive_radius>=c :
            break
        else:
            k = 1
            naive_radius += k
            up[0] = up[0]-k
            down[0] = down[0]+k
            left[1] = left[1]-k
            right[1] = right[1]+k
            
            diag_up_left[0] = diag_up_left[0]-k
            diag_up_left[1] = diag_up_left[1]-k
            
            diag_up_right[0] = diag_up_right[0]-k
            diag_up_right[1] = diag_up_right[1]+k
            
            diag_down_left[0] = diag_down_left[0]+k
            diag_down_left[1] = diag_down_left[1]-k
            
            diag_down_right[0] = diag_down_right[0]+k
            diag_down_right[1] = diag_down_right[1]+k
            
#     print("inner naive radius =",naive_radius)
    
    
#     #inner naive circle
#     for theta in range(0,361):
#         x = naive_radius*math.cos((theta/180)*math.pi) + palm_x
#         y = naive_radius*math.sin((theta/180)*math.pi) + palm_y
#         if x<r and y<c:
#             out_naive_circle[int(x)][int(y)] = 0
        
    #maximal inner radius
    while(True):
        naive_radius -= 1
        good_points = 0
        points_on_circle = []
        points_dictionary = dict()
        for theta in range(0,361):
            x = naive_radius*math.cos((theta/180)*math.pi) + palm_x
            y = naive_radius*math.sin((theta/180)*math.pi) + palm_y
            if points_dictionary.get(str(x)+"--"+str(y))==None:
                points_dictionary[str(x)+"--"+str(y)] = "Present"
                points_on_circle.append([x,y])
        for point in points_on_circle:
            all_neighbors_white = True
            a = point[0]
            b = point[1]
            up = [a-1,b]
            down = [a+1,b]
            left = [a,b-1]
            right = [a,b+1]
            diag_up_left = [a-1,b-1]
            diag_up_right = [a-1,b+1]
            diag_down_left = [a+1,b-1]
            diag_down_right = [a+1,b+1] 
            neighbors_list = [up,down,left, right, diag_up_left, diag_up_right, diag_down_left, diag_down_right]
            for neighbor in neighbors_list:
                if neighbor[0]>=256 or neighbor[1]>=c or out[int(neighbor[0])][int(neighbor[1])] == 0:
                    all_neighbors_white = False                    
                    break
            if all_neighbors_white == True:
                good_points += 1
        if good_points > 0.8*len(points_on_circle) or naive_radius <= 15:
            break
            
    max_inner_radius = naive_radius 
    
#     #maximal inner circle
#     for theta in range(0,361):
#         x = max_inner_radius*math.cos((theta/180)*math.pi) + palm_x
#         y = max_inner_radius*math.sin((theta/180)*math.pi) + palm_y
#         if x<r and y<c:        
#             out_circle[int(x)][int(y)] = 0   
            
    #outer cirle
    out_circle = np.copy(out) #copying   
    outer_radius = max_inner_radius*1.2
    for theta in range(0,361):
        x = outer_radius*math.cos((theta/180)*math.pi) + palm_x
        y = outer_radius*math.sin((theta/180)*math.pi) + palm_y
        if x<r and y<c:        
            out_circle[int(x)][int(y)] = 0   
    
    #plotting images
    figIter,ax = plt.subplots(1,2)
    figIter.set_size_inches(20,10) 
    plt.sca(ax[0])
    plt.axis("off")
    plt.title("Binary Image",fontsize="15")
    ax[0].imshow(out, cmap='gray')
    
    plt.sca(ax[1])
    plt.axis("off")
    plt.title("Outer Circle (1.2*R)",fontsize="15")
    ax[1].imshow(out_circle, cmap='gray')
    plt.show()
    print("-------------------------------------------------------------------------------------------------------")    
    print()
    print()

    #terminating loop
    batch_size -= 1
    if batch_size == 0:
        break #remove this piece of code to run all samples
