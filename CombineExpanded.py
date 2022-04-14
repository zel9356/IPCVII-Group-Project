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
import sys
import effects

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
    cv.imshow("Canny Edges", Canny)
    cv.waitKey(0)

    # morphology - thicken those lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(Canny, cv.MORPH_CLOSE, kernel)
    cv.imshow("Morphology Result", opening)
    cv.waitKey(0)

    # find shapes
    contours, hierarchy = cv.findContours(opening,
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # cv.drawContours(roi_cropped, contours, -1, (0, 255, 0), 3)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_item = sorted_contours[0]

    # cv.drawContours(roi_cropped, largest_item, -1, (255, 0, 0), 10)

    cv.imshow('Largest Object Extracted', roi_cropped)
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

"""
Puts image 1 into image 2
"""
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

"""
Roates image angle degrees
"""
#https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated
def main():


    if len(sys.argv) == 6 or len(sys.argv ) == 7:

        # command line inputs

        # image 1 location and name
        imageName1 = sys.argv[1]

        #image 2 location and name
        imageName2 = sys.argv[2]

        # desired artistic effect
        artEffect = sys.argv[3]

        # Rotation angle
        rotation = int(sys.argv[4])

        # filp (-1) if no flip
        flip = int(sys.argv[5])

        # read in both images
        img1 = cv.imread(imageName1)
        img2 = cv.imread(imageName2)

        # get height and width of each image
        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape

        # if either dim on orginal image is greater than 1000, scale down
        if h1 or w1 > 1000:
            # lets make hieght 500 pixels cause i said so
            scale_percent = int(h1/500)
            #scale_percent = 50  # percent of original size
            width = int(img1.shape[1] /scale_percent )
            height = int(img1.shape[0] / scale_percent )
            dim = (width, height)
            # resize image
            img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)
        if rotation != 0:
            img1 = rotate(img1, rotation)
        if flip != -1:
            if flip == 1 or 0:
                img1 = cv.flip(img1, flip)


        # lets make sure image 2 fits on screen
        if h2 > 1000:
            scale_percent = int(h2 / 1000)
            width = int(img2.shape[1] /scale_percent )
            height = int(img2.shape[0] /scale_percent )
            dim = (width, height)

            # resize image
            img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)

        # lets grab the largest "object" from image1
        result = imageOne(img1)

        # Put the result/object in image 2
        result = image2(result, img2)
        # save image if told to
        if artEffect == "pencil":
            result = effects.pencil(result)
        elif artEffect =="water":
            result = effects.watercolor(result)
        elif artEffect == "oil":
            result = effects.oilPainting(result)
        cv.imshow("Art Effect", result)
        cv.waitKey(0)
        if len(sys.argv) ==7:
            dst = sys.argv[6]
            cv.imwrite(dst, result)
    else:
        print("Usage: <image1 name> <image2 name> <artistic effect: oil, pencil, water> <rotation angle> <flip (-1: no flip, 0: horizontal, 1:vertical)> \n or <image1 name> <image2 name> <artistic effect: oil, pencil, water> <rotation angle> <flip (-1: no flip, 0: horizontal, 1:vertical)> <output file name>")



if __name__ == '__main__':
    main()
