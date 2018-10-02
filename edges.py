"""Uses the Canny edge detection method to detect and show the edges of an image. There are trackbars controlling
   the upper and lower thresholds for edge detection to illustrate their impact on edge detection sensitivity."""


import argparse
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--sigma')
args = vars(parser.parse_args())

image_path = args['filename']

img = cv2.imread(image_path)
# converting to grayscale preserves the important features in the image while eliminating unnecessary data
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurring reduces noise in the image. Using a larger kernel (e.g. (5x5)) increases blur
processed = cv2.GaussianBlur(grayscale, (3,3), 0)

# sigma is a value that controls the threshold size
sigma = args['sigma'] if args['sigma'] else 0.3
med_pixel_intensity = np.median(processed)
upper_limit = int(max(0, (1.0 - sigma) * med_pixel_intensity))
lower_limit = int(min(255, (1.0 - sigma) * med_pixel_intensity))
edges = cv2.Canny(processed, upper_limit, lower_limit)

cv2.imshow('Original', img)
cv2.imshow('Grayscale', grayscale)
cv2.imshow('Blurred', processed)
cv2.imshow('Edges', edges)
lower_pos = 0
upper_pos = 255


def print_val(val):
    """Updates window with new edge detection parameters."""
    lower = cv2.getTrackbarPos('Lower Threshold', 'Edges')
    upper = cv2.getTrackbarPos('Upper Threshold', 'Edges')
    new_edges = cv2.Canny(processed, upper, lower)
    cv2.imshow('Edges', new_edges)

cv2.createTrackbar('Lower Threshold', 'Edges', lower_pos, 255, print_val)
cv2.createTrackbar('Upper Threshold', 'Edges', upper_pos, 255, print_val)
cv2.waitKey(0)
