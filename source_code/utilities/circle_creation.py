import numpy as np

def completeCircle(circle_pixels, xc, yc, x, y, color=None, image=None):
    circle_pixels.append([xc+x, yc+y])
    circle_pixels.append([xc-x, yc+y])
    circle_pixels.append([xc+x, yc-y]) 
    circle_pixels.append([xc-x, yc-y]) 
    circle_pixels.append([xc+y, yc+x]) 
    circle_pixels.append([xc-y, yc+x]) 
    circle_pixels.append([xc+y, yc-x]) 
    circle_pixels.append([xc-y, yc-x])
    if not (color is None or image is None):
        x_max, y_max = image.shape
        if 0 <= xc+x < x_max and 0 <= yc+y < y_max:  
            image[xc+x, yc+y] = color
        if 0 <= xc-x < x_max and 0 <= yc+y < y_max:
            image[xc-x, yc+y] = color
        if 0 <= xc+x < x_max and 0 <= yc-y < y_max:
            image[xc+x, yc-y] = color
        if 0 <= xc-x < x_max and 0 <= yc-y < y_max:
            image[xc-x, yc-y] = color
        if 0 <= xc+y < x_max and 0 <= yc+x < y_max:
            image[xc+y, yc+x] = color
        if 0 <= xc-y < x_max and 0 <= yc+x < y_max:
            image[xc-y, yc+x] = color
        if 0 <= xc+y < x_max and 0 <= yc-x < y_max:
            image[xc+y, yc-x] = color
        if 0 <= xc-y < x_max and 0 <= yc-x < y_max:
            image[xc-y, yc-x] = color

    
def bresenhamCircle(xc, yc, r, color=None, image=None):
    circle_pixels = []
    x, y = 0, int(r)
    d = 3 -2*r
    is_target_pixel = False
    completeCircle(circle_pixels, xc, yc, x, y, color=color, image=image)
    while y >= x:
        #for each pixel draw eight point
        x+=1
        if d >= 0:
            y-=1
            d = d + 4*(x - y) + 10
        else:
            d = d + 4*x + 6
        completeCircle(circle_pixels, xc, yc, x, y, color=color, image=image)
    return np.array(circle_pixels)