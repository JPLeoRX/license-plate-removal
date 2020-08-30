from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy

imagePath = '../sample-images/3.jpg'
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
x,y,w,h = cv2.boundingRect(plateContour)
plateImage = imageCv[y:y+h, x:x+w]
plateImageBlur = cv2.GaussianBlur(plateImage, (25, 25), 0)

def findMostOccurringColor(cvImage) -> (int, int, int):
    width, height, channels = cvImage.shape
    colorCount = {}
    for y in range(0, height):
        for x in range(0, width):
            BGR = (int(cvImage[x, y, 0]), int(cvImage[x, y, 1]), int(cvImage[x, y, 2]))
            if BGR in colorCount:
                colorCount[BGR] += 1
            else:
                colorCount[BGR] = 1

    maxCount = 0
    maxBGR = (0, 0, 0)
    for BGR in colorCount:
        count = colorCount[BGR]
        if count > maxCount:
            maxCount = count
            maxBGR = BGR

    return maxBGR

plateBackgroundColor = findMostOccurringColor(plateImageBlur)
tempContours3 = cv2.drawContours(imageCv.copy(), [plateContour], -1, plateBackgroundColor, -1)
cv2.imshow('Original Image', imageCv)
cv2.imshow('Result', tempContours3)

cv2.waitKey()