## Goal:
Develop a Faster R-CNN model ancillary to 
[TimelapsEr](https://github.com/SamuelClucas/SC_TSL_09092024_TimelapsEr) 
and the [imaging 
system](https://github.com/SamuelClucas/SC_TSL_06082024_Imaging-System-Design) 
to handle identification of plates, labels, and colonies, by the system 
within the incubator.

#### Handling local dependencies and external dependency imports: 
All dirs inside src containing __init__.py will be installed as packages under their own namespace, facilitating imports. 
Before running scripts/*.py, use:  

``` 
pip install --upgrade pip
pip install -e .
```
Or, if not developing the project, just:  

```
pip install --upgrade pip
pip install .
```
