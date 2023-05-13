import numpy as np
import cv2

def preprocess(image):
    # Perform preprocess to the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = blurred.astype(np.uint8)
    # Calculate the gradient
    x1 = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    x2 = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(x1 ** 2 + x2 ** 2)
    # Determine the threshold by mean and std error
    threshold = np.mean(grad) + np.std(grad)
    # Adjust the image
    threshold, preprocessed = cv2.threshold(grad, threshold, 255, cv2.THRESH_BINARY)
    preprocessed = preprocessed.astype(np.uint8)
    return preprocessed

def FqdrLtl(contour, epsilon=0.02):
    # Find the quadrilateral
    eps = epsilon * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) == 4:
        return approx
    return None

def FLC(image):
    # Find the contour points of the quadrilateral
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_approx = None
    # Iterate to get four vertices
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            approx = FqdrLtl(cnt)
            if approx is not None:
                largest_area = area
                largest_approx = approx
    return largest_approx


def order_points(pts):
    # Ordering the endpoints of the quadrilateral
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    # top - left
    rect[0] = pts[np.argmin(s)]
    # top - right
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    # bottom - right
    rect[1] = pts[np.argmin(diff)]
    # bottom - left
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # Perform image transformation based on four points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Define the width and height
    max_width = min(sum((tl - tr) ** 2) ** 0.5, sum((bl - br) ** 2) ** 0.5)
    max_width = int(max_width)
    max_height = min(sum((tl - bl) ** 2) ** 0.5, sum((tr - br) ** 2) ** 0.5)
    max_height = int(max_height)
    # Define the corresponding transform matrix
    dst = np.array([
        [max_width * 0.1, max_height * 0.1],
        [max_width * 1.1 - 1, max_height * 0.1],
        [max_width * 1.1 - 1, max_height * 1.1 - 1],
        [max_width * 0.1, max_height * 1.1 - 1]], dtype="float32")
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # Warp the image according to the transform matrix
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    return warped


def imrect(im1):
    # Perform Image rectification on an 3D array im.
    # Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
    # Returns: out: numpy.ndarray, rectified image。
    #   out =im1
    # Preprocess the input image
    processed = preprocess(im1)
    # Find the contour we want
    largest_contour = FLC(processed)
    # Rectify the image
    rect = four_point_transform(im1, largest_contour.reshape(4, 2))
    out = rect
    return (out)

if __name__ == "__main__":

    # This is the code for generating the rectified output
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = np.array(cv2.imread(name, -1), dtype=np.float32)/255.
        image = image*255
        rectificated = imrect(image)/255.
        cv2.imwrite('./data/Result_'+name[7:],np.uint8(np.clip(np.around(rectificated*255,decimals=0),0,255)))


img = cv2.imread('path/to/image.jpg')