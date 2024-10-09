# Establishing a GoHREP for Analysis 0001\*

9/10/24

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

Next: [01_image_collection.md](01_image_collection.md)
