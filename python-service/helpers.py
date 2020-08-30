from typing import List

from PIL import Image as imageMain
from PIL.Image import Image
import cv2
import numpy

# This class contains all helper functions that should be reused
class Helpers():
    def openImagePil(self, imagePath: str) -> Image:
        return imageMain.open(imagePath)

    def convertPilImageToCvImage(self, pilImage: Image):
        return cv2.cvtColor(numpy.array(pilImage), cv2.COLOR_RGB2BGR)

    def openImageCv(self, imagePath: str):
        return self.convertPilImageToCvImage(self.openImagePil(imagePath))

    def cvToGrayScale(self, cvImage):
        return cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)

    def cvApplyBilateralFilter(self, cvImage):
        return cv2.bilateralFilter(cvImage, 11, 17, 17)

    def cvApplyGaussianBlur(self, cvImage, size: int):
        return cv2.GaussianBlur(cvImage, (size, size), 0)

    def cvToCannyEdge(self, cvImage):
        return cv2.Canny(cvImage, 170, 200)

    # Extracts all contours from the image, and resorts them by area (from largest to smallest)
    def cvExtractContours(self, cvImage):
        contours, hierarchy = cv2.findContours(cvImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        return contours

    # Find contours that look like rectangles
    def cvFilterRectangleContours(self, contours) -> List:
        rectangleContours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approximationAccuracy = 0.02 * perimeter
            approximation = cv2.approxPolyDP(contour, approximationAccuracy, True)
            if len(approximation) == 4:
                rectangleContours.append(contour)
        return rectangleContours

    # Crops an image around a given contour (returns the area inside this contour)
    def cvCropByContour(self, cvImage, contour):
        newImage = cvImage.copy()
        x,y,w,h = cv2.boundingRect(contour)
        return newImage[y:y+h, x:x+w]

    # Counts all unique BGR colors, and return the most occurring one
    def cvFindMostOccurringColor(self, cvImage) -> (int, int, int):
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

    # Finds a rough approximation of center point for a given contour
    def cvFindCenterPointOfContour(self, contour) -> (int, int):
        moments = cv2.moments(contour)
        centerPointX = int(moments["m10"] / moments["m00"])
        centerPointY = int(moments["m01"] / moments["m00"])
        return (centerPointX, centerPointY)

    # Given an existing contour resize it to a given ratio
    def cvResizeContour(self, contour, resizeRatio: float):
        centerPointX, centerPointY = self.cvFindCenterPointOfContour(contour)
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