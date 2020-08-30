from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy

imagePath = '../sample-images/1.jpg'
imagePil = imageMain.open(imagePath)
imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)
bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
blur = cv2.GaussianBlur(bilateral, (5, 5), 0)

edged = cv2.Canny(blur, 170, 200)
cv2.imshow('After Canny Edge', edged)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
tempContours1 = cv2.drawContours(imageCv.copy(), contours, -1, (255, 0, 0), 2)
cv2.imshow('Detected Contours', tempContours1)

cv2.waitKey()