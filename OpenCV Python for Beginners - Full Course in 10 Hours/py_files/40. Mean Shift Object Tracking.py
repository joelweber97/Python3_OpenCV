#!/usr/bin/env python
# coding: utf-8

# ### Mean Shift Object Tracking

# object tracking is the process of locating a moving object over time using a camera.

# The idea behind mean shift is to locate the densest point within a window. Then create a new window around that point. Then recalculate the densest point within that new window. Then create a new window around that point, etc, etc, until the the densest point/window location doesn't move.  (Similar concept to the idea behind k-means clustering)
# ![image.png](attachment:image.png)

# histogram back projection creates an image of the same size but of a single channel of our input image (frame) where each pixel corresponds to the probability of that pixel belonging to our object.

# In[36]:


import numpy as np
import cv2


# In[49]:


cap = cv2.VideoCapture('./data/highway.mp4')


# In[50]:


print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# In[51]:


#take the first frame of the video
ret, frame = cap.read()

#setup initial location of window
x,y,width,height = 460, 175, 50, 40
track_window = (x,y,width, height)

#define histogram back projection
#set up the ROI for tracking
roi = frame[y:y+height, x:x+width]
#convert roi to hsv color space
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#calculate mask
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
#calculate histogram values
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
#normalize hist values
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)


#setup the termination criteria, either 10 iterations or move by at least 1 point
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


while(1):
    ret,frame = cap.read()
    if ret == True:
        #convert the frame to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #calculate back projected image
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        #apply mean shift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        #draw it on image
        x,y,w,h = track_window
        final_image = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
        
        cv2.imshow('final_image', final_image)
        cv2.imshow('back projected', dst)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()


# Limitations:
# 1. The size of the window doesn't change.
# 2. We have to know the starting point for the window

# In[ ]:




