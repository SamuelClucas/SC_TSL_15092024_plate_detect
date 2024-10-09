#!/bin/bash

cd ~/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/images/mixed/
for f in *.png; do
    filename="$(basename $f .png)"
    ECHO $filename
    if [ -f ~/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/xml_labels/${filename}.xml ]; then # if f present at directory, then
      cp $f ../positives
    fi 
done

# script to copy labelled images to output directory (unused images left behind
