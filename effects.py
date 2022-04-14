"""
Artstic  effect code from https://towardsdatascience.com/painting-and-sketching-with-opencv-in-python-4293026d78b

applies effect to image to make it look more "art" like

Author: Zoe LaLena

Date: 4/14/2022

IPCV II Group Project
"""



import cv2


img = cv2.imread('img.jpg')

# gives image pencil sketch effect
def pencil(img):
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    # sigma_s and sigma_r are the same as in stylization.
    # shade_factor is a simple scaling of the output image intensity.
    # The higher the value, the brighter is the result. Range 0 - 0.
    return dst_color

#gives image watercolor effect
def watercolor(img):
    res = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    # sigma_s controls the size of the neighborhood. Range 1 - 200
    # sigma_r controls the how dissimilar colors within the neighborhood will be averaged.
    # A larger sigma_r results in large regions of constant color. Range 0 - 1
    return res

#gives image oil paining effect
def oilPainting(img):
    res = cv2.xphoto.oilPainting(img, 7, 1)
    return res




