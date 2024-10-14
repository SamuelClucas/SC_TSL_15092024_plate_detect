# Training ResNet Model from Local .pth File

7/10/24

#### Problem: 

After successfully configuring a singularity container on the cluster to
run
[train_positives_with_helpers.py](../../../scripts/model_implementation/train_positives_with_helpers.py)
(see overview [here](ResNet50_setup.md)), it became clear I would have
to train from a locally defined model. See the following error file:

``` {bash}
Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth" to /hpc-home/cla24mas/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
Traceback (most recent call last):
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 1350, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 1240, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 1286, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 1235, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 1006, in _send_output
    self.send(msg)
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 946, in send
    self.connect()
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 1402, in connect
    super().connect()
  File "/opt/software/.venv/lib/python3.8/http/client.py", line 917, in connect
    self.sock = self._create_connection(
  File "/opt/software/.venv/lib/python3.8/socket.py", line 808, in create_connection
    raise err
  File "/opt/software/.venv/lib/python3.8/socket.py", line 796, in create_connection
    sock.connect(sa)
OSError: [Errno 101] Network is unreachable

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "train_positives_with_helpers.py", line 38, in <module>
    model, preprocess = helper_training_functions.get_model_instance_object_detection(num_class)
  File "/hpc-home/cla24mas/SC_plate_detect/src/helper_training_functions.py", line 27, in get_model_instance_object_detection
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0001)
  File "/opt/software/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py", line 142, in wrapper
    return fn(*args, **kwargs)
  File "/opt/software/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py", line 228, in inner_wrapper
    return builder(*args, **kwargs)
  File "/opt/software/.venv/lib/python3.8/site-packages/torchvision/models/detection/faster_rcnn.py", line 659, in fasterrcnn_resnet50_fpn_v2
    model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
  File "/opt/software/.venv/lib/python3.8/site-packages/torchvision/models/_api.py", line 90, in get_state_dict
    return load_state_dict_from_url(self.url, *args, **kwargs)
  File "/opt/software/.venv/lib/python3.8/site-packages/torch/hub.py", line 765, in load_state_dict_from_url
    download_url_to_file(url, cached_file, hash_prefix, progress=progress)
  File "/opt/software/.venv/lib/python3.8/site-packages/torch/hub.py", line 624, in download_url_to_file
    u = urlopen(req)
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 222, in urlopen
    return opener.open(url, data, timeout)
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 525, in open
    response = self._open(req, data)
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 542, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 502, in _call_chain
    result = func(*args)
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 1393, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "/opt/software/.venv/lib/python3.8/urllib/request.py", line 1353, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 101] Network is unreachable>
srun: error: t384n7: task 0: Exited with exit code 1
```

This error arises from
[src.helper_training_functions.get_model_instance_object_detection()](../../../src/helper_training_functions.py),
defined as:

``` python
def get_model_instance_object_detection(num_class: int) -> fasterrcnn_resnet50_fpn_v2:
    # New weights with accuracy 80.858%
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT # alias is .DEFAULT suffix, weights = None is random initialisation, box MAP 46.7, params, 43.7M, GFLOPS 280.37 https://github.com/pytorch/vision/pull/5763
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0001)
    preprocess = weights.transforms()
    # finetuning pretrained model by transfer learning
    # get num of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
    return model, preprocess
```

**Breakdown:**  
- This approach requires permissions to download the model file from the
internet (through ‘torch.utils.model_zoo.load_url()’). For this reason,
it is not amenable to training on the cluster.  
- For this reason, I need to instead instantiate a ResNet model from a
local .py file defining the class and load model weights from a locally
installed .pth file.  

My plan is to follow [this
documentation](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/).
Pytorch provides ‘recipes’

See pytorch’s documentation on \[ResNet50 faster R-CNN
backbone\]\](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) -
(where by ‘backbone’ I mean the convolutional layer/module used to
extract features for the succeeding Region Proposal Network and
classification/bounding box regression heads). The documentation states
that it is based on the “[Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal
Networks](https://arxiv.org/abs/1506.01497)” paper, which is
reassuring.  