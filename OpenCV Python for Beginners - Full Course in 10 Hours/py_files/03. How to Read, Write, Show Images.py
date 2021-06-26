#!/usr/bin/env python
# coding: utf-8

# ### reading and saving images

# In[ ]:


import cv2
import numpy as np


# In[4]:


img = cv2.imread('./data/lena.jpg', -1)
cv2.imshow('img',img)
k = cv2.waitKey(0) & 0xFF
#if the escape key is hit it will close the window
#if the s key is hit it will save the image
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('./data/lena_copy.png', img)
    cv2.destroyAllWindows()


# In[ ]:




