# Dataset Creation

9/10/24

## GoHREP: 

### Goal: 

Collect images of growth plates inside an incubator using
[timelapser](https://github.com/SamuelClucas/SC_TSL_09092024_timelapser).
This image dataset will be used to train the CNN used by the [imaging
system](https://github.com/SamuelClucas/SC_TSL_06082024_imaging_system_design)
to detect and move to plates.  

### Hypothesis: 

Capturing images in a mock environment that mimics that in which the
final imaging system will be installed will allow me to establish a
proof-of-concept - the concept being that a CNN can be trained to
perform this task in this context.  

### Rationale: 

Object detection using model architectures like the [Faster
R-CNN](https://arxiv.org/pdf/1506.01497) has been robustly established
in the literature (see a recent review
[here](https://www.sciencedirect.com/science/article/pii/S1051200422004298)).  

### Experimental Plan: 

- Secure the RaspberryPi (RPi) and camera module inside an incubator
  using tape (given the physical system has not yet been created),
  feeding its power-cable through the incubator’s access port.  
- Collect several unused growth plates and stick on unused labels
  featuring some text.  
- Place the plates inside the incubator below the camera.  
- Using
  [timelapser](https://github.com/SamuelClucas/SC_TSL_09092024_timelapser),
  start a timelapse to take images at an appropriate interval to allow
  for random movements of the plates in between image captures. The
  flash (the led ring) indicates when an image capture is taking place.
  Whilst on, close the incubator door. Whilst off (deduced roughly from
  the specified imaging frequency), open the door and randomly
  redistribute the plates inside the incubator. Repeat until satisfied -
  bearing in mind the IMX 298 camera module connected to the RPi
  captures images in 4K. These images are to undergo preprocessing prior
  to labelling. This will include subsetting each image into evenly
  sized crops with dimensions roughly equivalent to that of a single
  plate from the camera’s POV. This means each image captured will
  ultimately result in n-images, where n = columns\*rows of the crop.  

## Image Collection: 

It is important that the steps taken to collect the images for the
training dataset are clearly outlined. Any experimental procedure is
likely to ingrain some bias within the dataset as a consequence of its
design. In its documentation, these biases can later be idenitified and
accounted/compensated for.  

### Setting up the camera inside the incubator: 

This is the incubator Simon Foster allowed me to use:  
![Incubator](../img/incubator.jpeg)  

I used cellotape to temporarily fix the RPi board to the side so that it
is beyond the camera’s field of view, the camera secured to the top
looking down.  
![Mock-setup](../img/setup.jpeg)  

### Using timelapser: 

Any appropriate image capture rate for the timelapse would work. As an
example:  

``` {bash}
#| eval: false

cd path/to/timelapser
source camera/bin/activate 
sudo camera/bin/python3 scripts/start_timelapse.py --units h --duration 1 --samples 120 --path path/to/desired/image/output/directory
```

This script starts a timelapse for 1 hour, taking an image every 30
seconds.  

Standard output stream (not a screenshot… apologies):  
![Example standard output stream from
timelapser](../img/timelapser.jpeg)  

This results in images like this:  
![Raw image capture](../img/raw_example.png)  

## Cropping and Downsizing Images: 

The libcamera_still_capture.sh script that drives the arducam module in
[timelapser](https://github.com/SamuelClucas/SC_TSL_09092024_timelapser)
takes 4K images. I used timelapser’s start_timelapse.py script to create
the image dataset to train models in this repo, hence my images are also
in 4K.  

As in the reference [paper](https://arxiv.org/pdf/1506.01497), I need to
scale down the dataset samples so that their shortest side is 600 pixels
in length.  

The following program crops the raw .png images down into composite
images and scales them down. For the sake of space and time efficiency
(as git struggles with large file uploads), I have included just one
[‘raw_example.png’](../img/raw_example.png) image.  

### Program Overview: 

#### Imports:

``` python
import cv2
import numpy as np
import time
from pathlib import Path
```

#### Main:

``` python
if __name__ == '__main__':
    inputPath = Path('path/to/raw/uncropped/images')
    outputPath = 'desired/output/path'

    for img in inputPath.iterdir():
        if not img.name.startswith('.') and img.name.endswith('.png'):
            print(str(img))
            image = cv2.imread(img)

            sizeY = image.shape[0]
            sizeX = image.shape[1]
            
            mCols = 4
            nRows = 4

            for i in range(nRows):
                for j in range(mCols):
                    roi = image[i*int(sizeY/nRows):i*int(sizeY/nRows) + int(sizeY/nRows) , j*int(sizeX/mCols):j*int(sizeX/mCols) + int(sizeX/mCols)]

                    roi_sizeY = roi.shape[0]
                    roi_sizeX =  roi.shape[1]

                    scale_down = 600/min(roi_sizeY, roi_sizeX)
                    cv2.imwrite(outputPath + str(i) + '_' + str(j) + '_' + img.name, cv2.resize(roi, None, fx = scale_down, fy = scale_down, interpolation = cv2.INTER_LINEAR))
                    print("file saved")
```

**Breakdown:**  
- Each raw image is cropped into 4x4 composite images (16 sub-images per
raw image), then scaled down so that their shortest side is 600 pixels
long, like these:

<img src="../img/downsized/0_0_raw_example.png" width="100"
alt="image" />
<img src="../img/downsized/0_1_raw_example.png" width="100"
alt="image" />
<img src="../img/downsized/0_2_raw_example.png" width="100"
alt="image" />
<img src="../img/downsized/0_3_raw_example.png" width="100"
alt="image" />  
*And so on…*  

- This is achieved by looping through each .png in the directory (here,
  just a single ‘raw_example.png’) in which another 2 for loops use
  [array
  slicing](https://www.w3schools.com/python/numpy/numpy_array_slicing.asp)
  to crop the raw image into its composite sub-sections.  
- Each crop is stored in ‘roi’ which is scaled down so that its shortest
  side is 600 pixels long.  
- Each roi is written to a csv file using
  [cv2.imwrite()](https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/).
  This way, the cropped image’s aspect ratio is maintained (final
  dimensions: 600x799).  
- Each crop is saved at the specified output path in the form
  ‘row_column_MMDDhhmmss.png’.  

After labelling plates in the images using LabelImg, I wrote a a simple
shell script to copy the images that had at least one plate labelled to
a new directory:

``` {bash}
#!/bin/bash

cd ../../raw/mixed/
for f in *.png; do
    filename="$(basename $f .png)"
    ECHO $filename
    if [ -f ~/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/xml_labels/${filename}.xml ]; then # if f present at directory, then
      cp $f ../positives
    fi 
done
```

This forms the basis of the [positive dataset](../raw/positives), where
there is at least one plate present in each image with a corresponding
xml label file.  

## Converting Labels from XML to CSV: 

I used [LabelImg](https://pypi.org/project/labelImg/) to label the image
‘samples’ with bounding boxes around any visible plates (or parts of
plates) in the sample. Hence for this dataset, there are two classes:
plate, background. This comprises the ‘positives’ dataset, i.e., each
sample has at least one bounding box label.  

> [!NOTE]
>
> ### LabelImg code modifications
>
> *Note:* as of when I used LabelImg, the installation was broken and
> required modifications to two files, as detailed below:
>
> Filename: canvas.py From: p.drawRect(left_top.x(), left_top.y(),
> rect_width, rect_height)
>
> To: p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width),
> int(rect_height))
>
> From: p.drawLine(self.prev_point.x(), 0, self.prev_point.x(),
> self.pixmap.height())
>
> To: p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()),
> int(self.pixmap.height()))
>
> From: p.drawLine(0, self.prev_point.y(), self.pixmap.width(),
> self.prev_point.y())
>
> To: p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()),
> int(self.prev_point.y()))
>
> Filename: labelImg.py From: bar.setValue(bar.value() +
> bar.singleStep() \* units)
>
> To: bar.setValue(int(bar.value() + bar.singleStep() \* units))
>
> Credit:https://github.com/HumanSignal/labelImg/issues/872#issuecomment-1402362685

I made the mistake of saving my image labels in PASCAL VOC format (as
.xml files). It’s convenient to read data from csv files when creating a
custom dataset class to be handled by pytorch’s
[DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
Hence, I had to store the bounding box vertices’ coordinates and class
label in a csv file.  

### Parsing XML: 

XML is a hierarchical data format, hence is best represented with a tree
comprised of nodes.  
[ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html)
provides an API for parsing XML data.  

A typical XML file generated from training data labelling in PASCAL VOC
format:

``` {xml}
<annotation>
    <folder>downsized_plate_subset</folder>
    <filename>0_0_0916161819.png</filename>
    <path>/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect-/Training_Data/Unlabelled/augmentation/downsized_plate_subset/0_0_0916161819.png</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>799</width>
        <height>600</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>plate</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>702</xmin>
            <ymin>306</ymin>
            <xmax>799</xmax>
            <ymax>600</ymax>
        </bndbox>
    </object>
</annotation>
```

### Program Overview:

#### Imports:

``` python
from pathlib import Path
import glob
import pandas as pd
import xml.etree.ElementTree as ET
```

#### Parsing xml input file:

``` python
def xml_to_csv(path):
    xml_list = []
    for xml_file in path.iterdir(): 
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('.'):
                value = (
                    root.find('filename').text,
                    int(root.find('size/depth').text),
                    int(root.find('size/width').text),
                    int(root.find('size/height').text),
                    root.find('object/name').text,
                    int(member.find('object/bndbox/xmin').text),
                    int(member.find('object/bndbox/ymin').text),
                    int(member.find('object/bndbox/xmax').text),
                    int(member.find('object/bndbox/ymax').text))
                xml_list.append(value)
            print(f"Processed {xml_file} successfully")
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
```

**Breakdown:**  
- using pathlib, but could equally use glob, a pattern that uses
wildcards to specify a set of files (e.g., the bounding box xml
files).  
- ‘Exception as e’ washelpful in debugging this code.  
- ET.parse() stores most of the elements in the xml file in tree nodes
(excluding XML comments, processing instructions, and doc type input
declarations), each with a tag and a dictionary of attributes accessed
by ‘node.tag’, ‘node.attrib’.  
- child nodes can be iterated over.  
- stores the data of interest in ‘xml_list’, which is returned to
main().  
- returns a pandas dataframe.

#### Main: 

``` python
def main():
    image_path = Path('path/to/xml/files')
    print(f"Searching for XML files in: {image_path}")
    xml_df = xml_to_csv(image_path)
    if xml_df is not None and not xml_df.empty:
        xml_df.to_csv('labels.csv', index=None)
        print('Successfully converted xml to csv.')
        print(f"CSV file contains {len(xml_df)} rows.")
    else:
        print("Failed to create CSV file. No data was extracted from XML files.")

if __name__ == '__main__':
    main()
```

**Breakdown:**  
- Uses
[pandas.DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
to write ‘xml_df’ to the ‘labels.csv’.  
- ‘index=None’ type=bool default is True, write row names.  

### Next step: create custom Plate_Image_Dataset class to be passed to Pytorch’s [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
