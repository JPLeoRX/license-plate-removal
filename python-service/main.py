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




imagePath = '/Users/leo/tekleo/license-plate-recognition/sample-images/1.jpg'
imagePil = openImage(imagePath)
print('width=' + str(imagePil.width) + ', height=' + str(imagePil.height))

image = toCvImage(imagePil)
gray = toGrayScale(image)
blur = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(blur, 170, 200)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]

plateContour = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        plateContour = contour
        break

plateMoments = cv2.moments(plateContour)
plateCenterPointX = int(plateMoments["m10"] / plateMoments["m00"])
plateCenterPointY = int(plateMoments["m01"] / plateMoments["m00"])
plateCenterPoint = (plateCenterPointX, plateCenterPointY)
print('plateMoments=' + str(plateMoments) + ', plateCenterPoint=' + str(plateCenterPoint))

#imageWithPlateContour = cv2.drawContours(image.copy(), [plateContour], -1, (0, 255, 0), -1)

plateContourResizedPoints = []
resizeRatio = 0.95
for i in range(0, len(plateContour)):
    x = plateContour[i][0][0]
    y = plateContour[i][0][1]

    x1 = x - plateCenterPointX
    y1 = y - plateCenterPointY

    x2 = x1 * resizeRatio
    y2 = y1 * resizeRatio

    x3 = x2 + plateCenterPointX
    y3 = y2 + plateCenterPointY

    resizedPoint = [x3, y3]
    plateContourResizedPoints.append(resizedPoint)
plateContourResized = numpy.array(plateContourResizedPoints, dtype=numpy.int32)

#imageWithPlateContour = cv2.drawContours(image.copy(), [plateContourResized], -1, (204, 184, 168), -1)
#imageWithPlateContour = cv2.circle(imageWithPlateContour, plateCenterPoint, 7, (255, 0, 0), -1)

#cv2.imshow('imageWithPlateContour', imageWithPlateContour)
#cv2.waitKey()

x,y,w,h = cv2.boundingRect(plateContourResized)
cropped = image[y:y+h, x:x+w]
croppedBlur = cv2.GaussianBlur(cropped, (9, 9), 0)
color = BackgroundColorDetector(croppedBlur).detect()
color = [int(c) for c in color]
colorReversed = (color[2], color[1], color[0])
print(colorReversed)

result = cv2.drawContours(image.copy(), [plateContourResized], -1, colorReversed, -1)
cv2.imshow('original', image)
cv2.imshow('result', result)
cv2.waitKey()

#print(color)