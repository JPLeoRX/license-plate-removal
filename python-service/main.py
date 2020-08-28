import cv2
import numpy
from PIL import Image as imageMain
from PIL.Image import Image

from detector import BackgroundColorDetector


def openImage(imagePath: str) -> Image:
    return imageMain.open(imagePath)

def toCvImage(pilImage: Image):
    return cv2.cvtColor(numpy.array(pilImage), cv2.COLOR_RGB2BGR)

def toPilImage(cvImage) -> Image:
    return imageMain.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))

def toGrayScale(cvImage):
    return cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)

def toBlurWithBilateralFilter(cvImage):
    return cv2.bilateralFilter(cvImage, 11, 17, 17)

def toBlurWithGaussian(cvImage, size: int):
    return cv2.GaussianBlur(cvImage, (size, size), 0)

def findRectangleContours(contours):
    rectangleContours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            rectangleContours.append(contour)
    return rectangleContours

def findCenterOfContour(contour) -> (int, int):
    moments = cv2.moments(contour)
    centerPointX = int(moments["m10"] / moments["m00"])
    centerPointY = int(moments["m01"] / moments["m00"])
    return (centerPointX, centerPointY)

def resizeContour(contour, resizeRatio: float):
    centerPointX, centerPointY = findCenterOfContour(contour)
    contourResizedPoints = []
    for i in range(0, len(contour)):
        x = contour[i][0][0]
        y = contour[i][0][1]

        x1 = x - centerPointX
        y1 = y - centerPointY

        x2 = x1 * resizeRatio
        y2 = y1 * resizeRatio

        x3 = x2 + centerPointX
        y3 = y2 + centerPointY

        resizedPoint = [x3, y3]
        contourResizedPoints.append(resizedPoint)
    return numpy.array(contourResizedPoints, dtype=numpy.int32)

def getBackgroundColorOfContour(image, contour):
    x,y,w,h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    croppedBlur = toBlurWithGaussian(cropped, 25)
    color = BackgroundColorDetector(croppedBlur).detect()
    color = [int(c) for c in color]
    colorReversed = (color[2], color[1], color[0])
    return colorReversed


imagePath = '/Users/leo/tekleo/license-plate-recognition/sample-images/5.jpg'
imagePil = openImage(imagePath)
image = toCvImage(imagePil)
gray = toGrayScale(image)
blur = toBlurWithBilateralFilter(gray)
#cv2.imshow('blur 1', blur)
blur = toBlurWithGaussian(blur, 5)
#cv2.imshow('blur 2', blur)
edged = cv2.Canny(blur, 170, 200)
#cv2.imshow('edged', edged)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
#t1 = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 2)
#cv2.imshow('t1', t1)

rectangleContours = findRectangleContours(contours)
#t2 = cv2.drawContours(image.copy(), rectangleContours, -1, (255, 0, 0), 2)
#cv2.imshow('t2', t2)
#cv2.waitKey()

# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
# rectangleContours = findRectangleContours(contours)
plateContour = rectangleContours[0]
plateContourResized = resizeContour(plateContour, 0.97)
plateBackgroundColor = getBackgroundColorOfContour(image, plateContourResized)
result = cv2.drawContours(image.copy(), [plateContourResized], -1, plateBackgroundColor, -1)
cv2.imshow('original', image)
cv2.imshow('result', result)
cv2.waitKey()

#print(color)

#cv2.waitKey()