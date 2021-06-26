#!/usr/bin/env python
# coding: utf-8

# ### Camshift Object Tracking

# Continuously Adaptive Mean Shift
# 
#  \
#  starts by applying mean shift but once the window converges it updates the size of the window

# In[1]:


import numpy as np
import cv2


# In[7]:


cap = cv2.VideoCapture('./data/highway.mp4')

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
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        #draw it on image
        pts = cv2.boxPoints(ret)
        #convert pts into integer
        pts = np.int0(pts)
        #draw the points
        final_image = cv2.polylines(frame, [pts], True, (255,0,255), 2)
        cv2.imshow('final_image', final_image)
        cv2.imshow('back projected', dst)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()


# This gives shitty results but I used a different image so I'm not surprized that the same params don't work for this video.

# In[ ]:




