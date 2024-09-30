#!/bin/bash

cd ~/Developer/SC_TSL_15092024_Plate_Detect/train/images/mixed/
for f in *.png; do
    filename="$(basename $f .png)"
    if [ ! -f ~/Developer/SC_TSL_15092024_Plate_Detect/train/xml_labels/${filename}.xml ]; then   
      cp $f ../negatives
    fi 
done

# script to copy labelled images to output directory (unused images left behind
