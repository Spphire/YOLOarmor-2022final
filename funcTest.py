import cv2
import numpy as np

img = cv2.imread('data/images/1-5.jpg')
bk = np.full((img.shape[0], img.shape[1], img.shape[2]), 114, dtype=np.uint8)
points=np.array([[120,100],[100,500],[500,520],[500,100]])
hull = cv2.convexHull(points)
cv2.polylines(img,[hull],True,(255,255,0),2)

mask = np.zeros([img.shape[0],img.shape[1],1], dtype=np.int8)
mask = cv2.fillPoly(mask, [hull], 255)
#mask = np.concatenate([mask,mask,mask],axis=2)
img = np.where(mask,img,bk)
cv2.imshow('img',img)
cv2.waitKey(0)
