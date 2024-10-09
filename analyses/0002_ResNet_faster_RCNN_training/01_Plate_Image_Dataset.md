# Plate_Image_Dataset Class Definition

8/9/25

### Purpose:

*Note:* I used
[this](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
Pytorch tutorial as a launchpoint. As per the tutorial, I downloaded
several helper functions available on Pytorch vision’s github.  

Why not just import them into scripts through the torchvision library?  
- See fmassa’s comment
[here](https://github.com/pytorch/vision/issues/2254).

If you install the package as described in the project
[README](../../README.md), they have already been installed under
[src/torchvision_deps/](../../src/torchvision_deps/) and can be imported
into scripts.  

For reference, I have included wget commands for each below:

``` {bash}
#| eval: false
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py 
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py
```

### Programme Overview:

#### Imports:

``` python
import os
from torchvision_deps import engine
import re
from glob import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from typing import Dict
```

#### Defining the constructor…

``` python
class Plate_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transforms=None):
        self.img_labels: pd.Dataframe = pd.read_csv(annotations_file) # bounding box vertices' coordinates
        self.boxes: np.array[int] = None

        self.img_files: list[str] = [f for f in glob(img_dir + "/*.png")]
        self.transforms = transforms
```

``` python
class Plate_Image_Dataset(Plate_Image_Dataset):
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[torch.Tensor, tv_tensors.Image]: 
        # loads images and bounding boxes
        img: torch.Tensor = read_image(self.img_files[idx]) # uint8
        
        # get digits from list object eg. ymaxs = '[ymax1, ymax2, ymax3]' returned from pd.DataFrame I KNOW IT'S HORRIBLE, IT'S THE ONLY WORKAROUND I FOUND WORKED :' (
        x1 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['xmins'][idx]))] #https://docs.python.org/3/library/re.html
        y1 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['ymins'][idx]))]
        x2 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['xmaxs'][idx]))]
        y2 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['ymaxs'][idx]))]

        num_objs = len(x1)
        labels = torch.ones((num_objs,), dtype=torch.int64) #int64 could be overkill, int8 since only one label, maybe downstream methods expect type   int64 so using for now as specified here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        
        # According to: https://pytorch.org/vision/stable/generated/torchvision.ops.masks_to_boxes.html#torchvision.ops.masks_to_boxes
        # masks_to_boxes returns [N, 4] Tensor x1, y1, x2, y2 where 0 <= x1 < x2 same for y1 and y2 [row = boxes, columns: x1y1x2y2]
        self.boxes = [[x1[i], y1[i], x2[i], y2[i]] for i in range(num_objs)]
        #for i in range(len(x1)):
        #    print(x1[i], '\n',y1[i], '\n', x2[i], '\n', y2[i], '\n', self.boxes)
       
        # one BoundingBoxes instance per sample, "{img": img, "bbox": BoundingBoxes(...)}" where BoundingBoxes contains all the bounding box vertices associated with that image in the form x1, y1,x2, y2
        self.boxes = torch.from_numpy(np.array(self.boxes)) # [N, 4] tensor, m x n matrix , Rows by cols
        
        image_id = idx
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (self.boxes[:, 3] - self.boxes[:, 1]) * (self.boxes[:, 2] - self.boxes[:, 0]) 
        # tv.tensors is tensor of images with associated metadata
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(self.boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #print(f'area is: {area}')
        if self.transforms is not None:
            img = self.transforms(img)
            target["boxes"] = self.transforms(target["boxes"])
        
        return img, target
```

**Breakdown:**  
- stepwise breakdown  

### Next step:
