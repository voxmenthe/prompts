*This model was released on 2022-04-26 and added to Hugging Face Transformers on 2025-01-08.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# ViTPose

[ViTPose](https://huggingface.co/papers/2204.12484) is a vision transformer-based model for keypoint (pose) estimation. It uses a simple, non-hierarchical [ViT](./vit) backbone and a lightweight decoder head. This architecture simplifies model design, takes advantage of transformer scalability, and can be adapted to different training strategies.

[ViTPose++](https://huggingface.co/papers/2212.04246) improves on ViTPose by incorporating a mixture-of-experts (MoE) module in the backbone and using more diverse pretraining data.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitpose-architecture.png)

You can find all ViTPose and ViTPose++ checkpoints under the [ViTPose collection](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335).

The example below demonstrates pose estimation with the [VitPoseForPoseEstimation](/docs/transformers/v4.56.2/en/model_doc/vitpose#transformers.VitPoseForPoseEstimation) class.


```
import torch
import requests
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, infer_device

device = infer_device()

url = "https://www.fcbarcelona.com/fcbarcelona/photo/2021/01/31/3c55a19f-dfc1-4451-885e-afd14e890a11/mini_2021-01-31-BARCELONA-ATHLETIC-BILBAOI-30.JPG"
image = Image.open(requests.get(url, stream=True).raw)

# Detect humans in the image
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]

# Human label refers 0 index in COCO dataset
person_boxes = result["boxes"][result["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

# Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

# Detect keypoints for each person found
image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
image_pose_result = pose_results[0]

xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

key_points = sv.KeyPoints(
    xy=xy, confidence=scores
)

edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=1
)
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.RED,
    radius=2
)
annotated_frame = edge_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=key_points
)
annotated_frame
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitpose.png)

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
# pip install torchao
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, TorchAoConfig

url = "https://www.fcbarcelona.com/fcbarcelona/photo/2021/01/31/3c55a19f-dfc1-4451-885e-afd14e890a11/mini_2021-01-31-BARCELONA-ATHLETIC-BILBAOI-30.JPG"
image = Image.open(requests.get(url, stream=True).raw)

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

inputs = person_image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]

person_boxes = result["boxes"][result["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge", device_map=device, quantization_config=quantization_config)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
image_pose_result = pose_results[0]
```

## Notes

* Use [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) to automatically prepare bounding box and image inputs.
* ViTPose is a top-down pose estimator. It uses a object detector to detect individuals first before keypoint prediction.
* ViTPose++ has 6 different MoE expert heads (COCO validation `0`, AiC `1`, MPII `2`, AP-10K `3`, APT-36K `4`, COCO-WholeBody `5`) which supports 6 different datasets. Pass a specific value corresponding to the dataset to the `dataset_index` to indicate which expert to use.


  ```
  from transformers import AutoProcessor, VitPoseForPoseEstimation, infer_device

  device = infer_device()

  image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
  model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base", device=device)

  inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(model.device)
  dataset_index = torch.tensor([0], device=device) # must be a tensor of shape (batch_size,)

  with torch.no_grad():
      outputs = model(**inputs, dataset_index=dataset_index)
  ```
* [OpenCV](https://opencv.org/) is an alternative option for visualizing the estimated pose.


  ```
  # pip install opencv-python
  import math
  import cv2

  def draw_points(image, keypoints, scores, pose_keypoint_color, keypoint_score_threshold, radius, show_keypoint_weight):
      if pose_keypoint_color is not None:
          assert len(pose_keypoint_color) == len(keypoints)
      for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
          x_coord, y_coord = int(kpt[0]), int(kpt[1])
          if kpt_score > keypoint_score_threshold:
              color = tuple(int(c) for c in pose_keypoint_color[kid])
              if show_keypoint_weight:
                  cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                  transparency = max(0, min(1, kpt_score))
                  cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
              else:
                  cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)

  def draw_links(image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width = 2):
      height, width, _ = image.shape
      if keypoint_edges is not None and link_colors is not None:
          assert len(link_colors) == len(keypoint_edges)
          for sk_id, sk in enumerate(keypoint_edges):
              x1, y1, score1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]), scores[sk[0]])
              x2, y2, score2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]), scores[sk[1]])
              if (
                  x1 > 0
                  and x1 < width
                  and y1 > 0
                  and y1 < height
                  and x2 > 0
                  and x2 < width
                  and y2 > 0
                  and y2 < height
                  and score1 > keypoint_score_threshold
                  and score2 > keypoint_score_threshold
              ):
                  color = tuple(int(c) for c in link_colors[sk_id])
                  if show_keypoint_weight:
                      X = (x1, x2)
                      Y = (y1, y2)
                      mean_x = np.mean(X)
                      mean_y = np.mean(Y)
                      length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                      angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                      polygon = cv2.ellipse2Poly(
                          (int(mean_x), int(mean_y)), (int(length / 2), int(stick_width)), int(angle), 0, 360, 1
                      )
                      cv2.fillConvexPoly(image, polygon, color)
                      transparency = max(0, min(1, 0.5 * (keypoints[sk[0], 2] + keypoints[sk[1], 2])))
                      cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                  else:
                      cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)

  # Note: keypoint_edges and color palette are dataset-specific
  keypoint_edges = model.config.edges

  palette = np.array(
      [
          [255, 128, 0],
          [255, 153, 51],
          [255, 178, 102],
          [230, 230, 0],
          [255, 153, 255],
          [153, 204, 255],
          [255, 102, 255],
          [255, 51, 255],
          [102, 178, 255],
          [51, 153, 255],
          [255, 153, 153],
          [255, 102, 102],
          [255, 51, 51],
          [153, 255, 153],
          [102, 255, 102],
          [51, 255, 51],
          [0, 255, 0],
          [0, 0, 255],
          [255, 0, 0],
          [255, 255, 255],
      ]
  )

  link_colors = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
  keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

  numpy_image = np.array(image)

  for pose_result in image_pose_result:
      scores = np.array(pose_result["scores"])
      keypoints = np.array(pose_result["keypoints"])

      # draw each point on image
      draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4, show_keypoint_weight=False)

      # draw links
      draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)

  pose_image = Image.fromarray(numpy_image)
  pose_image
  ```

## Resources

Refer to resources below to learn more about using ViTPose.

* This [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTPose/Inference_with_ViTPose_for_body_pose_estimation.ipynb) demonstrates inference and visualization.
* This [Space](https://huggingface.co/spaces/hysts/ViTPose-transformers) demonstrates ViTPose on images and video.

## VitPoseImageProcessor

### class transformers.VitPoseImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vitpose/image_processing_vitpose.py#L328)

( do\_affine\_transform: bool = True size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_affine\_transform** (`bool`, *optional*, defaults to `True`) —
  Whether to apply an affine transformation to the input images.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 256, "width": 192}`):
  Resolution of the image after `affine_transform` is applied. Only has an effect if `do_affine_transform` is set to `True`. Can
  be overridden by `size` in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to normalize the input with mean and standard deviation.
* **image\_mean** (`list[int]`, defaults to `[0.485, 0.456, 0.406]`, *optional*) —
  The sequence of means for each channel, to be used when normalizing images.
* **image\_std** (`list[int]`, defaults to `[0.229, 0.224, 0.225]`, *optional*) —
  The sequence of standard deviations for each channel, to be used when normalizing images.

Constructs a VitPose image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vitpose/image_processing_vitpose.py#L423)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] boxes: typing.Union[list[list[float]], numpy.ndarray] do\_affine\_transform: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **boxes** (`list[list[list[float]]]` or `np.ndarray`) —
  List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the bounding
  box coordinates in COCO format (top\_left\_x, top\_left\_y, width, height).
* **do\_affine\_transform** (`bool`, *optional*, defaults to `self.do_affine_transform`) —
  Whether to apply an affine transformation to the input images.
* **size** (`dict[str, int]` *optional*, defaults to `self.size`) —
  Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
  resizing.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use if `do_normalize` is set to `True`.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*, defaults to `'np'`) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

* **pixel\_values** — Pixel values to be fed to a model, of shape (batch\_size, num\_channels, height,
  width).

Preprocess an image or batch of images.

#### post\_process\_pose\_estimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vitpose/image_processing_vitpose.py#L597)

( outputs: VitPoseEstimatorOutput boxes: typing.Union[list[list[list[float]]], numpy.ndarray] kernel\_size: int = 11 threshold: typing.Optional[float] = None target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None  ) → `list[list[Dict]]`

Parameters

* **outputs** (`VitPoseEstimatorOutput`) —
  VitPoseForPoseEstimation model outputs.
* **boxes** (`list[list[list[float]]]` or `np.ndarray`) —
  List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the bounding
  box coordinates in COCO format (top\_left\_x, top\_left\_y, width, height).
* **kernel\_size** (`int`, *optional*, defaults to 11) —
  Gaussian kernel size (K) for modulation.
* **threshold** (`float`, *optional*, defaults to None) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will be resize with the default value.

Returns

`list[list[Dict]]`

A list of dictionaries, each dictionary containing the keypoints and boxes for an image
in the batch as predicted by the model.

Transform the heatmaps into keypoint predictions and transform them back to the image.

## VitPoseConfig

### class transformers.VitPoseConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vitpose/configuration_vitpose.py#L28)

( backbone\_config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None backbone: typing.Optional[str] = None use\_pretrained\_backbone: bool = False use\_timm\_backbone: bool = False backbone\_kwargs: typing.Optional[dict] = None initializer\_range: float = 0.02 scale\_factor: int = 4 use\_simple\_decoder: bool = True \*\*kwargs  )

Parameters

* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*, defaults to `VitPoseBackboneConfig()`) —
  The configuration of the backbone model. Currently, only `backbone_config` with `vitpose_backbone` as `model_type` is supported.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **scale\_factor** (`int`, *optional*, defaults to 4) —
  Factor to upscale the feature maps coming from the ViT backbone.
* **use\_simple\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether to use a `VitPoseSimpleDecoder` to decode the feature maps from the backbone into heatmaps. Otherwise it uses `VitPoseClassicDecoder`.

This is the configuration class to store the configuration of a [VitPoseForPoseEstimation](/docs/transformers/v4.56.2/en/model_doc/vitpose#transformers.VitPoseForPoseEstimation). It is used to instantiate a
VitPose model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the VitPose
[usyd-community/vitpose-base-simple](https://huggingface.co/usyd-community/vitpose-base-simple) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import VitPoseConfig, VitPoseForPoseEstimation

>>> # Initializing a VitPose configuration
>>> configuration = VitPoseConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = VitPoseForPoseEstimation(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VitPoseForPoseEstimation

### class transformers.VitPoseForPoseEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vitpose/modeling_vitpose.py#L194)

( config: VitPoseConfig  )

Parameters

* **config** ([VitPoseConfig](/docs/transformers/v4.56.2/en/model_doc/vitpose#transformers.VitPoseConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The VitPose model with a pose estimation head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vitpose/modeling_vitpose.py#L213)

( pixel\_values: Tensor dataset\_index: typing.Optional[torch.Tensor] = None flip\_pairs: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.vitpose.modeling_vitpose.VitPoseEstimatorOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **dataset\_index** (`torch.Tensor` of shape `(batch_size,)`) —
  Index to use in the Mixture-of-Experts (MoE) blocks of the backbone.

  This corresponds to the dataset index used during training, e.g. For the single dataset index 0 refers to the corresponding dataset. For the multiple datasets index 0 refers to dataset A (e.g. MPII) and index 1 refers to dataset B (e.g. CrowdPose).
* **flip\_pairs** (`torch.tensor`, *optional*) —
  Whether to mirror pairs of keypoints (for example, left ear — right ear).
* **labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

Returns

`transformers.models.vitpose.modeling_vitpose.VitPoseEstimatorOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.vitpose.modeling_vitpose.VitPoseEstimatorOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VitPoseConfig](/docs/transformers/v4.56.2/en/model_doc/vitpose#transformers.VitPoseConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Loss is not supported at this moment. See <https://github.com/ViTAE-Transformer/ViTPose/tree/main/mmpose/models/losses> for further detail.
* **heatmaps** (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`) — Heatmaps as predicted by the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VitPoseForPoseEstimation](/docs/transformers/v4.56.2/en/model_doc/vitpose#transformers.VitPoseForPoseEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, VitPoseForPoseEstimation
>>> import torch
>>> from PIL import Image
>>> import requests

>>> processor = AutoImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")
>>> model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]
>>> inputs = processor(image, boxes=boxes, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> heatmaps = outputs.heatmaps
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vitpose.md)
