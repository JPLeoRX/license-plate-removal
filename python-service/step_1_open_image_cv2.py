from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy

imagePath = '../sample-images/1.jpg'
imagePil = imageMain.open(imagePath)
imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)
cv2.imshow('Original Image', imageCv)
cv2.waitKey()