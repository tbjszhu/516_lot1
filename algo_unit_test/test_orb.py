import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

filename = "demo.png"
img = cv2.imread(filename)

# Initiate STAR detector
orb = cv2.ORB()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print des.shape
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()
cv2.imwrite( "demo_orb"+'.png', img2);
