from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy

imagePath = '../sample-images/1.jpg'
imagePil = imageMain.open(imagePath)
imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)

gray = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Scaled', gray)

bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow('After Bilateral Filter', bilateral)

blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
cv2.imshow('After Gausian Blur', blur)

cv2.waitKey()