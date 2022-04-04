"""

"""
import math

import numpy as np
import cv2 as cv
def imageOne(img):
    roi = cv.selectROI(img)

    roi_cropped = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    cv.imshow("crop", roi_cropped)
    cv.waitKey(0)
    # Convert to graycsale

    img_gray = cv.cvtColor(roi_cropped, cv.COLOR_BGR2GRAY)

    # Blur the image for better edge detection

    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    #sobelxy = cv.Sobel(src=img_blur, ddepth=-1, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    sobelxy = cv.Canny(img_blur, 50, 200)
    cv.imshow("edge", sobelxy)
    cv.waitKey(0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    #opening = cv.morphologyEx(sobelxy, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(sobelxy, cv.MORPH_CLOSE, kernel)
    cv.imshow("Morph", opening)
    cv.waitKey(0)



    minLineLength = 50
    maxLineGap = 100
    lines = cv.HoughLines(opening, 1, np.pi /180, 150, minLineLength, maxLineGap)
    blank = np.zeros(shape=opening.shape, dtype=np.uint8)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(blank, pt1, pt2, (255, 255, 255), 3, cv.LINE_AA)
    cv.imshow('Largest Object', blank)
    cv.waitKey(0)

    contours, hierarchy = cv.findContours(blank,
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #cv.drawContours(roi_cropped, contours, -1, (0, 255, 0), 3)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_item = sorted_contours[0]

    #cv.drawContours(roi_cropped, largest_item, -1, (255, 0, 0), 10)

    cv.imshow('Largest Object', roi_cropped)
    result_img = np.zeros(shape=roi_cropped.shape, dtype=np.uint8)

    # Create a mask image that contains the contour filled in
    hull = cv.convexHull(largest_item)
    row,col,channels = roi_cropped.shape
    for r in range(0,row):
        for c in range(0,col):
            result = cv.pointPolygonTest(largest_item, (c,r), False)
            if result == 1 or result ==0:
                result_img[r,c] = roi_cropped[r, c]

    cv.imshow('Largest Object', result_img)
    cv.waitKey(0)
    return result_img
def image2(img1, img2):
    roi = cv.selectROI(img2)
    img1 = cv.resize(img1, (roi[2], roi[3]), interpolation = cv.INTER_AREA)

    cv.imshow('Largest Object', img1)
    cv.waitKey(0)
    black = (0,0,0)
    row, col, channels = img1.shape
    r2, c2, channels2 = img2.shape
    for r in range(0, row):
        for c in range(0, col):
            if img1[r,c][0] == 0 and img1[r,c][1] == 0 and img1[r,c][2] == 0:
                pass
            else:

                img2[r+roi[1], c+roi[0]] = img1[r,c]

    cv.imshow('Largest Object', img2)
    cv.waitKey(0)

def main():
    img1 = cv.imread("b1.jpg")
    result  = imageOne(img1)

    img2 = cv.imread("desert.jpg")
    image2(result, img2)

if __name__ == '__main__':
    main()