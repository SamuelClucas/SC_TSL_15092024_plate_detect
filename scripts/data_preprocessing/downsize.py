import cv2
import numpy as np
import time
from pathlib import Path


if __name__ == '__main__':
    inputPath = Path('/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/images/raw_example/')
    outputPath = '/Users/cla24mas/Documents/My_Repos/SC_TSL_15092024_Plate_Detect/train/images/raw_example/downsized/'

    for img in inputPath.iterdir():
        if not img.name.startswith('.'):
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
                

