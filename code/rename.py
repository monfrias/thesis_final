# Author: Mon Cedrick G. Frias
# Date: May 11, 2018
# Filename: rename.py
# Description: This program changes names of every image
# Code Reference: https://stackoverflow.com/questions/9835243/python-rename-multiple-image-files

# Importing library
import os

# Declare variables
DRYRUN = False 

# Create a list of files from a specific directory
files = [ f for f in os.listdir('../dataset/training_set/non_coconuts') ]

# Scanning all files
for (index, filename) in enumerate(files):
    extension = os.path.splitext(filename)[1]
    newname = "20180521_coconut%03d%s" % (index,extension)

    # Check if there are name duplicates
    if os.path.exists(newname):
        print("Cannot rename %s to %s, already exists" % (filename,newname))
        continue
    
    # Dry run test only
    if DRYRUN:
        print("Would rename %s to %s" % (filename,newname))

    # Actual renaming of files
    else:
        os.rename("../dataset/training_set/non_coconuts/%s" % (filename), "../dataset/training_set/non_coconuts/%s" % (newname))
        print("Renaming %s to %s" % (filename,newname))

