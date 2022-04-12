"""
places objects from 1 image into another

usage: python3 <image1> <image2>

Author: Zoe LaLena

Date: 4/12/2022

IPCV II Group Project
"""

import math
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
"""
extract the biggest shape from image1
img: image 1 to extract an object from
"""
def imageOne(img):

    # select bounding box
    roi = cv.selectROI(img)

    # crop based on bounding box
    roi_cropped = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    cv.imshow("crop", roi_cropped)
    cv.waitKey(0)


    # Convert to graycsale
    img_gray = cv.cvtColor(roi_cropped, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    # Canny edge detection
    Canny = cv.Canny(img_blur, 100, 200)
    cv.imshow("edge", Canny)
    cv.waitKey(0)

    # morphology - thicken those lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(Canny, cv.MORPH_CLOSE, kernel)
    cv.imshow("Morph", opening)
    cv.waitKey(0)

    # find shapes
    contours, hierarchy = cv.findContours(opening,
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # cv.drawContours(roi_cropped, contours, -1, (0, 255, 0), 3)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_item = sorted_contours[0]

    # cv.drawContours(roi_cropped, largest_item, -1, (255, 0, 0), 10)

    cv.imshow('Largest Object', roi_cropped)
    result_img = np.zeros(shape=roi_cropped.shape, dtype=np.uint8)

    # Create a mask image that contains the contour filled in
    hull = cv.convexHull(largest_item)
    row, col, channels = roi_cropped.shape
    for r in range(0, row):
        for c in range(0, col):
            result = cv.pointPolygonTest(largest_item, (c, r), False)
            if result == 1 or result == 0:
                result_img[r, c] = roi_cropped[r, c]

    cv.imshow('Largest Object', result_img)
    cv.waitKey(0)
    return result_img


def image2(img1, img2):
    roi = cv.selectROI(img2)
    img1 = cv.resize(img1, (roi[2], roi[3]), interpolation=cv.INTER_AREA)

    cv.imshow('Largest Object', img1)
    cv.waitKey(0)
    black = (0, 0, 0)
    row, col, channels = img1.shape
    r2, c2, channels2 = img2.shape
    for r in range(0, row):
        for c in range(0, col):
            if img1[r, c][0] == 0 and img1[r, c][1] == 0 and img1[r, c][2] == 0:
                pass
            else:

                img2[r + roi[1], c + roi[0]] = img1[r, c]

    cv.imshow('Largest Object', img2)
    cv.waitKey(0)
    return img2


def main():


    img1 = cv.imread("apple.png")

    scale_percent = 25  # percent of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)

    result = imageOne(img1)



    # image 2
    img2 = cv.imread("desert.jpg")
    result = image2(result, img2)
    # save image
    cv.imwrite("result.jpg", result)



if __name__ == '__main__':
    main()
