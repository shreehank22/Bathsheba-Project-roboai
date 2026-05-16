import cv2
import numpy as np


def segment_red(rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask1= cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv_image, (170, 100, 100), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)


    # binary mask of the red color
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours 
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask,None,False
    largest = max(contours,key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:  
        return mask,None,False

    # centroid
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return mask,None,False
    u = int(M['m10'] / M['m00'])
    v = int(M['m01'] / M['m00'])

    return mask,(u,v),True
