import cv2
import cv
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('test.png')

# Create SURF object. Set Hessian Threshold to 400
#surf = cv2.SURF(400)
#surf = cv2.SURF()
img = cv.LoadImage('test.png')
img_gray = cv.CreateImage(cv.GetSize(img), img.depth, 1)
'''storage = cv.CreateMemStorage()
kp, des = cv.ExtractSURF(img_gray, None, storage, (1,100,4,1))'''

hessian_threshold = 5000
surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img,None)
#detector = cv2.SURF(hessian_threshold)
#kp,des = detector.detectAndCompute(img_gray,None)


# Find keypoints and descriptors directly
#kp, des = surf.detectAndCompute(img,None)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()
