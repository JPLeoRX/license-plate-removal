from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy

def test(imagePath):
    imagePil = imageMain.open(imagePath)
    imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
    edged = cv2.Canny(blur, 170, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]

    rectangleContours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximationAccuracy = 0.02 * perimeter
        approximation = cv2.approxPolyDP(contour, approximationAccuracy, True)
        if len(approximation) == 4:
            rectangleContours.append(contour)

    plateContour = rectangleContours[0]
    tempContours2 = cv2.drawContours(imageCv.copy(), [plateContour], -1, (255, 0, 0), 2)
    cv2.imshow('Detected Plate Contour ' + imagePath, tempContours2)

test('../sample-images/2.jpg')
test('../sample-images/3.jpg')
cv2.waitKey()