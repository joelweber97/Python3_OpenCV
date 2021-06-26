#!/usr/bin/env python
# coding: utf-8

# ### Detect Corners with Shi Tomasi Corner Detector

# In[1]:


import cv2
import numpy as np


# In[26]:


#read in image and convert it to grayscale
img = cv2.imread('./data/pic1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# The Shi Tomasi Detector is similar to the Harris Corner Detector except a difference in the way R is calculated. This method gives a better result than Harris Corner Detector \
#  \
# We can also define how many corners we want returns which can be useful 

# In[27]:


#pass the gray image into the Shi Tomasi goodFeaturesToTrack method
#goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
#only 25 corners are detected. Set this to 0 to detect all the corners.
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
print(corners)
#returns a array of arrays in float format.


# In[28]:


#since the goodFeaturesToTrack returns an array of floats we need to convert it to an array of ints
corners = np.int0(corners)
print(corners)


# In[31]:


#iterate throught the corners
for i in corners:
    #find values of x and y
    x,y = i.ravel()
    #draw a filled in circle with on the x,y cordinates with a 3 pixel radius
    cv2.circle(img, (x,y), 3, 255, -1)


# In[30]:


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# With the number of corners set to 25 and quality level set to 0.01 and minDist to 10 we get
# ![image.png](attachment:image.png)

# When we set max number of corners to 0 it will detect all the corners with a quality level above 0.01 and minDist = 10
# ![image.png](attachment:image.png)

# Also when we set max num of corners to 0 and quality level to 0.5 and minDist = 10 we get fewer corners yet
# ![image.png](attachment:image.png)

# In[ ]:





# Comparison between Shi Tomasi detector and Harris detector
# ![image.png](attachment:image.png)
# Shi Tomasi gives better results and we can control the number of corners

# In[ ]:




