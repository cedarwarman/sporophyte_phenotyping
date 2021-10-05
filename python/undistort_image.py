#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import glob
import yaml
import os

# Load calibration values
calibration_path = os.path.join(os.path.dirname(__file__),
                                '../data/calibration.yaml')

with open(calibration_path) as f:
    loaded_dict = yaml.load(f, Loader=yaml.SafeLoader)
    print("Calibration loaded")

mtx = np.asarray(loaded_dict.get('camera_matrix'))
dist = np.asarray(loaded_dict.get('dist_coeff'))

# Undistorting the images
images = glob.glob('*.jpg')

for fname in images:
    print("Undistorting: " + fname)
    img = cv.imread(fname)
    h, w = img.shape[:2]

    # Alpha 0
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image (seems to be unnecessary with alpha of 0 above, and 
    # not doing this fixes the loss of resolution problem)
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
 
    # Write
    output_name = str(os.path.splitext(fname)[0]) + '_undistorted.jpg'
    cv.imwrite(output_name, dst)

