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

### Class Overview:

#### Imports…

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
from collections.abc import Sequence # for type hints like 'tuple[]': https://docs.python.org/3/library/typing.html
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

**Breakdown:**  
- ‘Plate_Image_Dataset’ is a custom dataset that represents a map from
keys/indices to data samples, hence inherits from PyTorch’s provided
abstract class
‘[torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)’.  
- Initialise ‘self.img_labels’ as as a [pandas
DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html),
reading from the csv file containing bounding box vertex coordinates
passed to the constructor as a string in ‘annotations_file’.  
- Declare ‘self.boxes’ as a [numpy
array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
of integers, later to be used to store all bounding boxes associated
with an image (using an index) in **getitem**.  
- Initialise ‘self.img_files’ as a list of strings, using
[glob](https://docs.python.org/3/library/glob.html) to find all images
(pathnames ending in .png) at the image dataset directory passed to the
constructor as a string in ‘img_dir’. The notation \[f for f in…\] is a
concise way to create lists in python (see [list
comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions)).  
- Initialise ‘self.transforms’ as transforms passed the constructor. The
default is None. See PyTorch’s transforms documentation
[here](https://pytorch.org/vision/master/transforms.html#transforms).  

#### Defining **len**…

``` python
class Plate_Image_Dataset(Plate_Image_Dataset):
    def __len__(self) -> int:
        return len(self.img_files)
```

- Just returns the number of image files in the dataset as an integer.  

#### Defining **getitem**…

``` python
class Plate_Image_Dataset(Plate_Image_Dataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tv_tensors.Image]: 
        # loads images and bounding boxes
        img: torch.Tensor = read_image(self.img_files[idx]) # uint8
        
        # get digits from self.img_labels eg. ymaxs = '[ymax1, ymax2, ymax3]' returned from pd.DataFrame
        x1 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['xmins'][idx]))]
        y1 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['ymins'][idx]))]
        x2 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['xmaxs'][idx]))]
        y2 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['ymaxs'][idx]))]

        num_objs = len(x1)
        # this class is for the positives dataset, and so every image has at least one labelled bounding box, hence should be tensor of ones shape [N]
        labels = torch.ones((num_objs,), dtype=torch.int64) 

        # boxes expected shape [N, 4] Tensor x1, y1, x2, y2 where 0 <= x1 < x2 same for y1 and y2 [row = boxes, columns: x1y1x2y2]
        self.boxes = [[x1[i], y1[i], x2[i], y2[i]] for i in range(num_objs)]
       
        # one BoundingBoxes instance per sample, "{img": img, "bbox": BoundingBoxes(...)}" where BoundingBoxes contains all the bounding box vertices associated with that image in the form x1, y1,x2, y2
        self.boxes = torch.from_numpy(np.array(self.boxes)) # [N, 4] tensor
        
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
        
        if self.transforms is not None:
            img = self.transforms(img)
            target["boxes"] = self.transforms(target["boxes"])
        
        return img, target
```

**Breakdown:**  
- I’m not particularly familiar with indexing pandas DataFrames. I used
pythons [regular expression
operations](https://docs.python.org/3/library/re.html) to extract digits
(‘\d’) from the dataframe at a given index. Using isdigit() doesn’t work
(for example, iterating through ‘200’ with isdigit() would return ‘2’,
‘0’, ‘0’). I struggled to use pandas ‘loc’ indexing attribute. I may
return to this to do so.  
- **getitem** must return a tuple as specified in the tutorial. Here,
img is of subclass of
[tv.Tensor](https://pytorch.org/vision/master/tv_tensors.html) ([an
image stored as a 3D
matrix/tensor](https://discuss.pytorch.org/t/what-is-image-really/151290)).
This tensor should be of shape \[3, H, W\], where 3 is the number of
channels. Target is a dict with the following fields:  
- “boxes”: another
[tv.Tensor](https://pytorch.org/vision/master/tv_tensors.html) subclass
with shape \[N, 4\] where N is the number of bounding boxes associated
with the image at the index being ‘got’. Columns are x1, y1, x2, y2,
where 0 \<= x1 \< x2 (same for y1 and y2). This is specified by
‘format’.  
- “labels”: class labels (here 0 for background, 1 for ‘plate’) of type
int64.  
- “image_id”: image index in dataset of type int64.  
- “area”: float torch.Tensor of shape \[N\]. The area of the bounding
box. This is used by coco_eval in
[src/torchvision_deps](../../src/torchvision_deps/) to separate metric
scores for small, medium and large boxes.  
- “iscrowd”: int64 torch.Tensor of shape \[N\]. Instances with
iscrowd=True are ignored during evaluation. This doesn’t really serve a
purpose other than to prevent errors popping up when using the
[torchvision_deps](../../src/torchvision_deps/). I plan to either
extricate it from the dependencies so that it can be removed from the
dataset class. For now, it is always initialised as a tensor of zeros
using
[torch.zeros](https://pytorch.org/docs/stable/generated/torch.zeros.html).  
- wrapping image tensors and bounding boxes in TVTensor subclasses
allows for application of torchvision [built-in
transformations](https://pytorch.org/vision/stable/transforms.html).
Based on the TVTensor subclass wrapping the object, the tranforms
dispatch the object to the appropriate implementation (as described
[here](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#what-are-tvtensors)).  
- in the tutorial, img and target are passed to trainsforms together and
returned as a tuple. For some reason, I could not get it to work this
way. I suspect it is something to do with the types of the fields of
target not all being TVTensors. Regardless, I pass boxes and img to
transforms separately, then return them from **getitem** as a tuple.
Hopefully this isn’t behaving unexpectedly in the torchvision deps -
unit tests necessary to confirm this.  

### Next step: [creating helper functions module for import into training/evaluation scripts…](02_helper_training_functions.md)