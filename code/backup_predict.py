# Author: Mon Cedrick G. Frias
# Date: May 11, 2018
# Filename: predict.py
# Code Reference: This program makes new predictions using the classification model and selective search versions

# Part 2.5 - Check the binary classification of coconuts and non-coconuts
from keras.preprocessing.image import ImageDataGenerator

# Perform Image Augmentation to training data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generate augmented training data set
training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Declare variable to check classes
binary_classification = training_set.class_indices

print "Classes:"
print binary_classification

# Part 3 - Making new predictions (single prediction)
import numpy as np
import cv2
import sys
from keras.models import load_model

# Load the model
model = load_model('../models/coconuts_pretrained_15.h5')

# #Open file for saving statistics
# file = open("../stats_new_ss.txt", "a")

# Read image file
newname = "sp_02"
img = cv2.imread("../dataset/predictions/images/" + newname + ".png")

# Open text file for segment dimensions
fname = "segments_02"
f = open("../gt_segments/" + fname + ".txt")

# Map all ground truth segments
groundTruthDimensionArray = []
for line in f.read().split():
    arr = line.split(',')
    arr = map(int, arr)
    groundTruthDimensionArray.append([arr[0], arr[1], arr[2]+arr[0], arr[3]+arr[1]])

print groundTruthDimensionArray
# Create threads for optimization (optional)
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

# Selective Segmentation
gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
gs.setBaseImage(img)

# Choose one selective search version to use
selSearch_type = raw_input('Choose one type: \ns - single strategy\nf - fast selSearch\nq - quality selSearch\nChoice: ')

# Single strategy
if selSearch_type == 's':
    gs.switchToSingleStrategy()

# Selective Search Fast
elif selSearch_type == 'f':
    gs.switchToSelectiveSearchFast()

# Selective Search Quality
elif selSearch_type == 'q':
    gs.switchToSelectiveSearchQuality()

# Otherwise, exit the program
else:
    print(__doc__)
    sys.exit(1)

# Get the total number of segments found in the image
rects = gs.process()

# Set your customized number of segments (~ 2000 for better accuracy)
if len(rects) < 2000:
    nb_rects = len(rects)
else:
    nb_rects = 2000

# Make a copy of the original image
wimg = img.copy()

# Declare variables
tp = 0
tn = 0
fp = 0
fn = 0
dimensionArray = []

# Inspect every segment
for i in range(len(rects)):
    if (i < nb_rects):
        x, y, w, h = rects[i]
        cropped = wimg[y: y+h, x: x+w]

        # Filter bounding box in terms of dimension (e.g. too wide, too long, etc.)
        if float(h/w) >= 1.5 or float(w/h) >= 1.5:
            tn += 1
            continue
        # Filter bounding box in terms of area (e.g. should be 96X96 pixels and up)
        elif h < 96 or w < 96:
            tn += 1
            continue
        else:
            # Change all segment dimensions to 64X64.
            resized = cv2.resize(cropped, (64, 64)) 

            # Convert image into an array with a 4th dimension
            test_image = np.expand_dims(resized, axis = 0)

            # Perform predictions
            result = model.predict(test_image)
            #print "Result:", result
            # If a segment is classified as coconuts, draw a bounding rectangle and update coconut count
            if result[0][0] == 0:
                IoUArray = []

                # Check if the current box is overlapping to ground truth boxes
                for l in groundTruthDimensionArray:
                    dx = min(l[2], x+w) - max(l[0], x)
                    dy = min(l[3], y+h) - max(l[1], y)

                    if (dx>0) and (dy>0):
                        # Update variables
                        overlapArea = dx * dy
                        detectedArea = w * h
                        groundTruthArea = (l[2] - l[0]) * (l[3] - l[1])
                        IoU = float(overlapArea) / ((groundTruthArea + detectedArea) - overlapArea)
                        IoUArray.append(IoU)

                    else:
                        # Otherwise, it does not overlap with ground truth: a true negative.
                        IoUArray.append(0.0)
                
                # Otherwise, it does not overlap with ground truth: a true negative.
                if len(IoUArray) == 0:
                    IoUArray.append(0.0)

                print "IoU: ", IoUArray
                maxIoU = max(IoUArray)
                index = IoUArray.index(maxIoU)
                IoUCount = 0
                print "Index: ", index

                # Inspect all IoU values and if a value reaches at least 50%, it is a prospect coconut object
                for i in IoUArray:
                    if i > 0.50:
                        IoUCount += 1

                #print "Count: ", IoUCount

                # Check if count exactly 1 (it means that the segment is correctly fitted to one ground truth coconut), add it to the list
                if IoUCount == 1:
                    dimensionArray.append([x, y, x+w, y+h])
                    del groundTruthDimensionArray[index]

            # Otherwise, detected segment is not a coconut. Ignore.
            else:
                #cv2.rectangle(wimg, (x, y), (x+w, y+h), (0, 0, 255), 1, cv2.LINE_AA)
                tn += 1

# Box all the target objects
for d in dimensionArray:
    cv2.rectangle(wimg, (d[0], d[1]), (d[2], d[3]), (255, 0, 0), 1, cv2.LINE_AA)
    tp += 1
    cv2.putText(wimg, "#{}".format(tp), (int((d[2] + d[0]) / 2 - 10), int((d[3] + d[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show statistics
print "|===== Choice: " + selSearch_type + " =====|"
print "Name:", newname
print "Detected segments (TP, TN, FP, FN):", len(rects)
print "Coconut count (TP):", tp
print "Overlapped coconuts (FP):", fp
print "Non-coconut count (TN):", tn
print "Overlap + non-coconut count:", (tn + fp)

# Write the statistics to a file
# file = open("../stats_pretrained_15.txt", "a")
# file.write("|===== Choice: " + selSearch_type + " =====|\n")
# file.write("Name: " + newname + "\n")
# file.write("Detected segments (TP, TN, and FP): " + str(len(rects)) + "\n")
# file.write("Coconut count (TP and/or FP): " +  str(tp) + "\n")
# file.write("Overlapped coconuts (TN1): " +  str(fp) + "\n")
# file.write("Non-coconut count (TN2): " + str(tn) + "\n")
# file.write("Overlap + non-coconut count (TN1+TN2): " + str(tn + fp) + "\n\n")
# Save output image
# Single strategy
if selSearch_type == 's':
    cv2.imwrite("../dataset/predictions/" + newname + ".png", wimg)
    print "Successfully saved single strategy file"
elif selSearch_type == 'f':
    cv2.imwrite("../dataset/predictions/" + newname + ".png", wimg)
    print "Successfully saved selective search fast file"
elif selSearch_type == 'q':
    cv2.imwrite("../dataset/predictions/" + newname + ".png", wimg)
    print "Successfully saved selective search quality file"

# #Close file
# file.close()

# Delete model to free memory
del model

