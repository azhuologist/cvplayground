"""Produces scans of documents from input pictures. Can optionally output OCR-read text from the scan."""


import argparse
import cv2
import numpy as np
import skimage.filters
import pytesseract
from PIL import Image


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """A wrapper for cv2's resize function to allow more flexibility.
       Source: https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py"""
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def detect_edges(image):
    """Uses the Canny edge detection algorithm to find the edges of an image."""
    processed = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, (3,3), 0)

    sigma = 0.3
    med_pixel_intensity = np.median(processed)
    upper_limit = int(max(0, (1.0 - sigma) * med_pixel_intensity))
    lower_limit = int(min(255, (1.0 - sigma) * med_pixel_intensity))
    edges = cv2.Canny(processed, upper_limit, lower_limit)
    return edges


def find_document_contour(edged_image):
    """Finds the contour matching a document (assumed to be a piece of paper) in an edge detected image.
       Raises: ValueError if no such contour could be found."""
    # find contours from an edge-detected image, and choose the ones with the largest area
    contours = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for cont in contours:
        arc_length = cv2.arcLength(cont, True)
        approx_curve = cv2.approxPolyDP(cont, 0.02 * arc_length, True)

        if len(approx_curve) == 4:
            print(approx_curve)
            return approx_curve

    raise ValueError()


def rectangular_perpective_transform(image, points):
    """Given 4 points in an image, returns the image with perspective warped to focus on the pixels
       enclosed by the given points."""
    # We first order our points so they go clockwise from the top left. Top left point must have the
    # lowest coordinate sum, while the bottom right must have the largest
    ordered_pts = np.empty((4, 2), dtype = 'float32')
    pt_sum = np.sum(points, axis = 1)
    ordered_pts[0] = points[np.argmin(pt_sum)]
    ordered_pts[2] = points[np.argmax(pt_sum)]

    # the top right should have smallest coordinate difference, bottom left the largest
    pt_diff = np.diff(points, axis = 1)
    ordered_pts[1] = points[np.argmin(pt_diff)]
    ordered_pts[3] = points[np.argmax(pt_diff)]

    # for convenience, we store the points as variables for convenience in calculating width / height
    (top_left, top_right, bottom_right, bottom_left) = ordered_pts

    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    width = int(max(top_width, bottom_width))

    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)
    height = int(max(left_height, right_height))

    # create destination coordinate points to give us a top-down view of the subimage enclosed by the original points
    dest_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype = 'float32')
    transform_matrix = cv2.getPerspectiveTransform(ordered_pts, dest_points)
    return cv2.warpPerspective(image, transform_matrix, (width, height))


def main():
    """Runs the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--ocr', action='store_true')
    args = vars(parser.parse_args())

    # After loading in the image, we do some resizing for performance
    image = cv2.imread(args['filename'])
    original = image.copy()
    ratio = image.shape[0] / 500.0
    image = resize(image, height = 500)

    edges = detect_edges(image)
    doc_contour = find_document_contour(edges)
    doc_outline = edges.copy()
    cv2.drawContours(doc_outline, [doc_contour], -1, (128, 255, 0), 3)

    # applies perspective transform and segments the document from the rest of image
    result_image = rectangular_perpective_transform(original, doc_contour.reshape(4,2) * ratio)
    print()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    threshold = skimage.filters.threshold_local(result_image, 11, offset = 10, method = "gaussian")
    result_image = (result_image > threshold).astype('uint8') * 255

    cv2.imshow("Original", resize(original, height = 650))
    cv2.imshow("Edges", edges)
    cv2.imshow("Contour Outline", doc_outline)
    cv2.imshow("Scanned (Low Quality Version)", resize(result_image, height = 650))
    cv2.waitKey(0)

    output_filename = args['filename'].split('.')[0] + '_scan.' + args['filename'].split('.')[1]
    cv2.imwrite(output_filename, result_image)

    if args['ocr']:
        text = pytesseract.image_to_string(Image.open(output_filename))
        print(text)


if __name__ == "__main__":
    main()
