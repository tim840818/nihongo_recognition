import cv2 as cv
import numpy as np
IMG_SIZE = (84, 84)

def im_clean(img, thresh = 40):
    ## remove noices
    img = cv.medianBlur(img, 3)
    ret, img = cv.threshold(img, thresh, 255, cv.THRESH_TOZERO)
    img = cv.GaussianBlur(img, (3,3), 0)
    ret, img = cv.threshold(img, thresh, 255, cv.THRESH_TOZERO)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, -1)
    # img = cv.GaussianBlur(img,(9,9),0)
    # ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)
    # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
    img = cv.erode(img, kernel, iterations=1)
    return img

def im_bin(img, thresh = 40):
    ret, img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    return img

def im_reshape(img, img_rows, img_cols, pad=10, tol=5): # reshape the handwriting part into 64x64, in a 84x84 img. The surrounding border is consist of zeroes.
    ## construct 84x84 frame, reshape the handwrting part into 64x64
    fig = np.zeros((img_rows, img_cols), dtype=np.uint8)
    ## get the column cropping boundary
    c_axis = img.sum(axis=0) # get sum intensity of each column
    c_axis = np.nonzero(c_axis)[0] # crop out the zero columns
    # c_min, c_max = c_axis[0], c_axis[-1]
    c_min, c_max = max(0, c_axis[0] - tol), min(img.shape[1], c_axis[-1] + tol) # add 5-pixel tolerance to the cropping border
    ## do the same thing for row cropping boundary
    r_axis = img.sum(axis=1)
    r_axis = np.nonzero(r_axis)[0]
    # r_min, r_max = r_axis[0], r_axis[-1]
    r_min, r_max = max(0, r_axis[0] - tol), min(img.shape[0], r_axis[-1] + tol)
    ## crop the handwriting part from the image
    img = img[r_min:r_max, c_min:c_max]
    ## resize the handwriting part into 64x64
    img = cv.resize(img, (img_rows - 2*pad, img_cols - 2*pad))
    ## put the handwriting part into the 84x84 frame with padding
    fig[pad : img_rows-pad, pad:img_rows-pad] = img
    return fig