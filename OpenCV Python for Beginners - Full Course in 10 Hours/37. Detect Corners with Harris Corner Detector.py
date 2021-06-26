#!/usr/bin/env python
# coding: utf-8

# ### Detect Corners with Harris Corner Detector

# a corner is a region in the image wiht large variation in intensity in all directions

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[ ]:





# In[1]:


import cv2
import numpy as np


# In[15]:


#read in image and convert it to 32 bit gray float
#cornerHarris method needs a 32 bit float grayscale image
img = cv2.imread('./data/chessboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)


# cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) \
#  \
# img: input image, it should be grayscale and float32 type \
#  \
# blocksize: it is the size of the neighborhood considered for corner detection \
#  \
# ksize: aperture parameter of sobel derivitive used. \
#  \
# k: harris detector free parameter in the equation.

# In[16]:


#plug the float32 gray image in with a blocksize, ksize, and k value
#cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])
dst = cv2.cornerHarris(gray, 2,3,0.04)


# In[17]:


#dilate the dst for some reason
dst = cv2.dilate(dst, None)


# In[18]:


#revert back to the original image with optimal threshold value and marking all
#corners with red color
img[dst > 0.01 * dst.max()] = [0,0,255]


# In[19]:


cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ![image.png](attachment:image.png)

# In[ ]:




