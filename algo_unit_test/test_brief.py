import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test1.png',0)

# Initiate STAR detector
#star = cv2.FeatureDetector_create("STAR")
maxSize = 16
star = cv2.StarDetector()
#for opencv3
#star = cv2.xfeatures2D.StarDetector_create()
#star.detect(img)

# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")

# find the keypoints with STAR
kp = star.detect(img)
#kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print brief.getInt('bytes')
print des.shape

img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
