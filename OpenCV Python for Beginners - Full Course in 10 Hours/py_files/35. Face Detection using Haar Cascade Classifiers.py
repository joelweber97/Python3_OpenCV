#!/usr/bin/env python
# coding: utf-8

# ### Face Detecting using Haar Cascade Classifiers Image

# Effective object detection method. A machine learning based approach where a cascade function is trained for a lot of positive and negative images. \
#  \
# First, a classifier (namely a cascade of boosted classifiers workign with haar-like features) is trained with a few hundred sample views of a patricular object (i.e. a face or a car), called positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary images of the same size. \
#  \
# After the classifier is trained it can be applied to a region of interest of an input image and if it outputs a 1 there's a face otherwise it will output a 0

# OpenCV comes with a trainer as well as a detector so you can train your own classifier to detect any object, like a watch or a car. \
#  \
# Opencv's github page also has some trained classifiers in xml format. \
#  \
# Since we want to detect faces we will use the haarcascade_fonrtalface_default.xml classifier

# In[1]:


import cv2
import numpy as np


# In[2]:


#before loading the image we will define the face classifier
face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')


# In[7]:


#read in the image and since the cascade classifier will work with grayscale image we need to
#convert it to gray
img = cv2.imread('./data/face4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[8]:


#use the detectMultiScale method on the face_cascade CascadeClassifier object
#detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]])
faces = face_cascade.detectMultiScale(gray, 
                                      scaleFactor = 1.1,
                                     minNeighbors = 4)


# objects = cv2.CascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbors) \
#  \
# objects: Vector of rectangles where each rectangle contains the detected object. The retangles may be partially outisde the original image \
#  \
# image: matrix of the type CV_8U containing an image where objects are detected. \
#  \
#  scaleFactor: parameter specifying how much the image size is reduced at each image scale. \
#   \
#  minNeighbors: param specifying how many neighbors each candidate rectangle should have to retain it. \

# In[9]:


#iterate over all the faces we have detected and draw our retangles
for (x,y,w,h) in faces:
    #rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)


# In[10]:


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Works very well
# ![image.png](attachment:image.png)

# In[ ]:





# ### Face Detecting using Haar Cascade Classifiers Video

# In[11]:


#same as doing it for an image we have to do it for every single frame of the video


# In[20]:


import cv2
import numpy as np

#before loading the image we will define the face classifier
face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')
 
import cv2

cap = cv2.VideoCapture(0)
#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #use the detectMultiScale method on the face_cascade CascadeClassifier object
    #detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]])
    faces = face_cascade.detectMultiScale(gray, 
                                          scaleFactor = 1.1,
                                         minNeighbors = 6)

    #iterate over all the faces we have detected and draw our retangles
    for (x,y,w,h) in faces:
        #rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)
    
    cv2.imshow('face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




