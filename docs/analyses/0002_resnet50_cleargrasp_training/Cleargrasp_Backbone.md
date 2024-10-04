# Cleargrasp_Backbone.py Class

4/10/24

### Purpose:

The following defines a class used to instantiate neural networks based
on google’s [cleargrasp](https://sites.google.com/view/cleargrasp). Its
pre-trained backbone may be particularly suited to the recognition of
transparent microbe plates. I believe a faster R-CNN with a cleargrasp
pre-trained backbone will outperform one with a
[COCO](https://cocodataset.org/#home) pre-trained resnet50 backbone
given the former’s feature maps should in principle identify regions of
an input image with the traits of a transparent object (for example,
glare, or refraction of light through the object).
[Here](https://github.com/Shreeyak/cleargrasp) is the code.  
- Google wrote the model using pytorch, however it is not
‘out-of-the-box’ configured for bounding box-style object detection; but
rather comprises three modules used for object masking, occusion
boundary detection, and identification of surface normals. For this
reason, I want to extract the backbone only, freeze its pre-trained
weights, and then configure successive RPN modules and classification
heads.

``` python
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from lib.sync_batch_norm import SynchronizedBatchNorm2d
```

**Breakdown:**  
- The class defined in succeeding code blocks is derived from the
available code
[here](https://github.com/Shreeyak/cleargrasp/blob/master/pytorch_networks/masks/modeling/backbone/resnet.py),
and is in itself a [resnet-based
architecture](https://www.run.ai/guides/deep-learning-for-computer-vision/pytorch-resnet).
Residual blocks avoid the problem of vanishing/exploding gradients
problematic during backpropagation seen with VGG-based architectures as
the number of layers present in the model increase.  
- The [default pytorch training ‘recipe’ for
ResNet50](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)
resulted in a top-1 accuracy of 80.9%.  

#### Bottleneck Layer:

``` python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

**Breakdown:**  
- Firstly, a bottleneck class is defined. Bottleneck layers may play
numerous distinct roles in a neural networks, but fundamentally they
serve to constrain dataflow between network components by downsizing
with respect to the number of nodes in the bottleneck layer relative to
the preceding layer. Why is this useful? For example, think of two
images which are identical in every way but their resolution - one is
1080p, the other 4K. Both images are essentially similarly visually
comprehensible, yet there are far more pixels in the 4K image. The only
difference is microscopic detail. From this, it could be suggested that
the pixels that comprise a 4k image contain less ‘information’ as far as
understanding the contents of the image, relative to the lower
resolution duplicate. They both successfully communicate the semantics
of the image, where one does so more ‘concisely’. Further to this,
passing a lower-resolution image with far fewer pixel (eg:250x250)
through a network is far more computationally efficient in terms of time
complexity by reducing network parameters as much as possible without
compromising feature extraction.  

#### ResNet Backbone:

*Note:* this is almost identical to the class provided
[here](https://github.com/Shreeyak/cleargrasp/blob/master/pytorch_networks/masks/modeling/backbone/resnet.py),
the only modification being that the pretrained model is not downloaded
from the [pytorch models
site](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
directly within the code due to HPC constraints. It is instead loaded
from a local .pth file in a models directory. - The .pth file download
link: https://download.pytorch.org/models/resnet50-11ad3fa6.pth  

``` python
class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, model_dir, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model(model_dir)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_dir):
        pretrain_dict = torch.load(model_dir)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
```

#### Construct ResNet-101 Model:

``` python
def ResNet101(model_dir, output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, model_dir, pretrained=pretrained)
    return model
```

**Breakdown:**  
- The model’s weights are initially set to those trained on the
[ImageNet1K](https://www.image-net.org/) dataset, version 2.  
- Provided within the repo is the artificial image set on which
cleargrasp was trained to reliably identify transparent objects. After
this training, the backbone weights are frozen.

#### Main:

Below outlines how an object of this class is instantiated. Make sure to
download the model’s .pth file first:

``` {bash}
curl -O https://download.pytorch.org/models/resnet50-11ad3fa6.pth 
```

Then change ‘model_dir’ string to path/to/your/model/.pth/file

``` python
if __name__ == "__main__":
    import torch
    model_dir = 'resnet50-11ad3fa6.pth'
    model = ResNet101(model_dir, BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
```

    torch.Size([1, 2048, 64, 64])
    torch.Size([1, 256, 128, 128])

**Next step:** write a script to instantiate a model and load dataset
before training.
