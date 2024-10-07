# Cropping and Downsizing

8/9/25

### Purpose:

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
[‘raw_example.png’](images/raw_example.png) image.  

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
    inputPath = Path('/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/images/raw_example/')
    outputPath = '/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/images/raw_example/downsized/'

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

<img src="images/downsized/0_0_0916161659.png" width="250"
alt="image" />
<img src="images/downsized/0_1_0916161726.png" width="250"
alt="image" />
<img src="images/downsized/0_2_raw_example.png" width="250"
alt="image" />  
And so on…  

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
[shell script](delete_unlabelled.sh) to copy the images that had at
least one plate labelled to a new directory. This forms the basis of the
[positive dataset](../../../train/images/positives), where there is at
least one plate present in each image with a corresponding xml label
file.

### Next step: convert xml label files into single csv file. See [xml_to_csv.py](xml_to_csv.md).