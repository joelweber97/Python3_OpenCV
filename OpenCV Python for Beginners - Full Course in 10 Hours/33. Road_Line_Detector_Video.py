#!/usr/bin/env python
# coding: utf-8

# ### Road Line Detection Pt 3

# In[ ]:


#doing line detection on a video is the same as an image except we just have to run it
#on every single frame of the image


# In[2]:


import numpy as np
import cv2


# In[3]:


#function that draws the lines
def draw_lines(img, lines):
    img = np.copy(img)
    #create blank image that matches the original image size
    blank_image = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype = np.uint8)
    #loop through line vectors
    for i in lines:
        #get the 4 coordinates for each line vector
        for x1,y1,x2,y2 in i:
            #draw the lines onto the blank image
            cv2.line(blank_image, (x1,y1), (x2,y2), (255,0,0), 4)
    
    #merge image with lines with the original image
    #addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


# In[4]:


#function that will mask out areas we don't want.
def region_of_interest(img, vertices):
    #create 0's in the same size as the img
    mask = np.zeros_like(img)
    #set the masked area to white
    match_mask_color = 255
    #sets the area inside the vertices to all white pixels.
    #fillPoly(img, pts, color[, lineType[, shift[, offset]]])
    cv2.fillPoly(mask, vertices, match_mask_color)
    #crop out the masked part using the bitwise_and function.
    #keeps the area were the images are different which would be the white area.
    #bitwise_and(src1, src2[, dst[, mask]])
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# In[16]:


def process(image):
    #find the edges
    #cvtColor(src, code[, dst[, dstCn]])
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
    canny_image = cv2.Canny(gray_image, 100, 200)
    #we need to run this before masking the image or the edge detection will pick up the edges
    #on the edges of the mask as well.

    #print(image.shape)

    #define the vertices of the polygon that we want to keep
    region_vertices = np.array([(110, 540), (440,320), (530, 320), (960, 540)], np.int32)

    #now we will crop the image
    cropped_image = region_of_interest(canny_image, [region_vertices])

    #use HoughLinesP() to get the lines
    #HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
    #this returns all the line vectors that are detected
    lines = cv2.HoughLinesP(cropped_image, 
                            rho = 2, 
                            theta = np.pi/60, 
                            threshold = 50, 
                            lines = np.array([]),
                            minLineLength = 20, 
                            maxLineGap = 100)

    #this returns all the line vectors that are detected
    #print(lines)

    #actually draw the lines on the original image using the draw_lines function create above
    image_with_lines = draw_lines(image, lines)
    return image_with_lines


# In[17]:


#read the video
cap = cv2.VideoCapture('./data/solidWhiteRight.mp4')
frame_time = 100
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        #run the process function on each frame.
        #this function will return the image with the lines for each frame
        frame = process(frame)
        cv2.imshow('frame',frame)
        #cv2.waitKey(x) where x = number of seconds between a frame
        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




