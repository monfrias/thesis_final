# Author: Mon Cedrick G. Frias
# Date: May 28, 2018
# Filename: ground_truth.py
# Code Reference: This program maps ground truth segments in an image.


import numpy as np
import cv2

DRYRUN = True
name = "sp_12"

# Read image file
img = cv2.imread("../dataset/predictions/images/" + name + ".jpg")

# Open text file for segment dimensions
f = open('../gt_segments/segments_12.txt')

# Map all segments within the image
for line in f.read().split():
    arr = line.split(',')
    arr = map(int, arr)
    cv2.rectangle(img, (arr[0], arr[1]), (arr[2]+arr[0], arr[3]+arr[1]), (255, 0, 0), 1, cv2.LINE_AA)

# View image with ground truths
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Don't save if dry run test only.
if DRYRUN:
    cv2.imwrite("../dataset/predictions/ground_truth/" + name + ".png", img)
    print "Done saving file " + name + ".png"