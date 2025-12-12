*This model was released on 2025-03-24 and added to Hugging Face Transformers on 2025-06-27.*

# EoMT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

[The Encoder-only Mask Transformer]((https://www.tue-mps.org/eomt)) (EoMT) model was introduced in the CVPR 2025 Highlight Paper *[Your ViT is Secretly an Image Segmentation Model](https://huggingface.co/papers/2503.19108)* by Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus.
EoMT reveals Vision Transformers can perform image segmentation efficiently without task-specific components.

The abstract from the paper is the following:

*Vision Transformers (ViTs) have shown remarkable performance and scalability across various computer vision tasks. To apply single-scale ViTs to image segmentation, existing methods adopt a convolutional adapter to generate multi-scale features, a pixel decoder to fuse these features, and a Transformer decoder that uses the fused features to make predictions. In this paper, we show that the inductive biases introduced by these task-specific components can instead be learned by the ViT itself, given sufficiently large models and extensive pre-training. Based on these findings, we introduce the Encoder-only Mask Transformer (EoMT), which repurposes the plain ViT architecture to conduct image segmentation. With large-scale models and pre-training, EoMT obtains a segmentation accuracy similar to state-of-the-art models that use task-specific components. At the same time, EoMT is significantly faster than these methods due to its architectural simplicity, e.g., up to 4x faster with ViT-L. Across a range of model sizes, EoMT demonstrates an optimal balance between segmentation accuracy and prediction speed, suggesting that compute resources are better spent on scaling the ViT itself rather than adding architectural complexity.*

This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali).
The original code can be found [here](https://github.com/tue-mps/eomt).

## Architecture Info

The `EoMT` model uses a DINOv2-pretrained Vision Transformer with **register tokens** as its backbone. EoMT simplifies the segmentation pipeline by relying solely on the encoder, eliminating the need for task-specific decoders commonly used in prior approaches.

Architecturally, EoMT introduces a small set of **learned queries** and a lightweight **mask prediction module**. These queries are injected into the final encoder blocks, enabling **joint attention** between image patches and object queries. During training, **masked attention** is applied to constrain each query to focus on its corresponding region—effectively mimicking cross-attention. This constraint is gradually phased out via a **mask annealing strategy**, allowing for **efficient, decoder-free inference** without compromising segmentation performance.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/eomt_architecture.png)

The model supports semantic, instance, and panoptic segmentation using a unified architecture and task-specific post-processing.

## Usage Examples

Use the Hugging Face implementation of EoMT for inference with pre-trained models.

### Semantic Segmentation

The EoMT model performs semantic segmentation using sliding-window inference. The input image is resized such that the shorter side matches the target input size, then it is split into overlapping crops. Each crop is then passed through the model. After inference, the predicted logits from each crop are stitched back together and rescaled to the original image size to get the final segmentation mask.

> **Note:**  
> If you want to use a custom target size for **semantic segmentation**, specify it in the following format:  
>  `{"shortest_edge": 512}`  
> Notice that `longest_edge` is not provided here — this is intentional. For semantic segmentation, images are typically **scaled so that the shortest edge is greater than or equal to the target size** hence longest\_edge is not necessary.


```
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/ade20k_semantic_eomt_large_512"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the segmentation mask
plt.imshow(preds[0])
plt.axis("off")
plt.title("Semantic Segmentation")
plt.show()
```

### Instance Segmentation

The EoMT model performs instance segmentation using padded inference. The input image is resized so that the longer side matches the target input size, and the shorter side is zero-padded to form a square. The resulting mask and class logits are combined through post-processing (adapted from Mask2Former) to produce a unified instance segmentation map, along with segment metadata like segment id, class labels and confidence scores.

> **Note:**  
> To use a custom target size, specify the size as a dictionary in the following format:  
>  `{"shortest_edge": 512, "longest_edge": 512}`  
> For both instance and panoptic segmentation, input images will be **scaled and padded** to this target size.


```
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/coco_instance_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_instance_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the segmentation mask
plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Instance Segmentation")
plt.show()
```

### Panoptic Segmentation

The EoMT model performs panoptic segmentation using the same padded inference strategy as in instance segmentation. After padding and normalization, the model predicts both thing (instances) and stuff (amorphous regions) classes. The resulting mask and class logits are combined through post-processing (adapted from Mask2Former) to produce a unified panoptic segmentation map, along with segment metadata like segment id, class labels and confidence scores.


```
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/coco_panoptic_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the panoptic segmentation mask
plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Panoptic Segmentation")
plt.show()
```

## EomtImageProcessor

### class transformers.EomtImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt.py#L247)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_normalize: bool = True do\_split\_image: bool = False do\_pad: bool = False image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None ignore\_index: typing.Optional[int] = None num\_labels: typing.Optional[int] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the input to a certain `size`.
* **size** (`int`, *optional*, defaults to 640) —
  Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
  sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
  the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size * height / width, size)`.
* **resample** (`int`, *optional*, defaults to `Resampling.BILINEAR`) —
  An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
  `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
  `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
  to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the input to a certain `scale`.
* **rescale\_factor** (`float`, *optional*, defaults to `1/ 255`) —
  Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to normalize the input with mean and standard deviation.
* **do\_split\_image** (`bool`, *optional*, defaults to `False`) —
  Whether to split the input images into overlapping patches for semantic segmentation. If set to `True`, the
  input images will be split into patches of size `size["shortest_edge"]` with an overlap between patches.
  Otherwise, the input images will be padded to the target size.
* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
* **image\_mean** (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`) —
  The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
* **image\_std** (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`) —
  The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
  ImageNet std.
* **ignore\_index** (`int`, *optional*) —
  Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
  denoted with 0 (background) will be replaced with `ignore_index`.
* **num\_labels** (`int`, *optional*) —
  The number of labels in the segmentation map.

Constructs a EoMT image processor. The image processor can be used to prepare image(s) and optional targets
for the model.

This image processor inherits from [BaseImageProcessor](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BaseImageProcessor) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt.py#L505)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[list[dict[int, int]], dict[int, int], NoneType] = None instance\_id\_to\_semantic\_id: typing.Optional[dict[int, int]] = None do\_split\_image: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None do\_pad: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None ignore\_index: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image or batch of images to preprocess.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The corresponding semantic segmentation maps with the pixel-wise annotations.
* **instance\_id\_to\_semantic\_id** (`list[dict[int, int]]` or `dict[int, int]`, *optional*) —
  A mapping between object instance ids and class ids.
* **do\_split\_image** (`bool`, *optional*, defaults to `self.do_split_image`) —
  Whether to split the input images into overlapping patches for semantic segmentation.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the input images.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Target size as a dictionary with `"shortest_edge"` and `"longest_edge"` keys.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use when resizing.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the input images by `rescale_factor`.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Factor to scale image pixel values.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the input images.
* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean for normalization. Single value or list for each channel.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation for normalization. Single value or list for each channel.
* **ignore\_index** (`int`, *optional*) —
  Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
  denoted with 0 (background) will be replaced with `ignore_index`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be `"pt"`, `"tf"`, `"np"`, or `"jax"`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  Channel format of the output image. Either `"channels_first"` or `"channels_last"`.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  Channel format of the input image.

Preprocesses images or a batch of images.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt.py#L818)

( outputs target\_sizes: list size: typing.Optional[dict[str, int]] = None  )

Post-processes model outputs into final semantic segmentation prediction.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt.py#L907)

( outputs target\_sizes: list threshold: float = 0.5 size: typing.Optional[dict[str, int]] = None  )

Post-processes model outputs into Instance Segmentation Predictions.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt.py#L850)

( outputs target\_sizes: list threshold: float = 0.8 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 stuff\_classes: typing.Optional[list[int]] = None size: typing.Optional[dict[str, int]] = None  )

Post-processes model outputs into final panoptic segmentation prediction.

## EomtImageProcessorFast

### class transformers.EomtImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt_fast.py#L104)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.eomt.image\_processing\_eomt\_fast.EomtImageProcessorFastKwargs]  )

Constructs a fast Eomt image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt_fast.py#L161)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Optional[list[torch.Tensor]] = None instance\_id\_to\_semantic\_id: typing.Optional[dict[int, int]] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.eomt.image\_processing\_eomt\_fast.EomtImageProcessorFastKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The segmentation maps to preprocess for corresponding images.
* **instance\_id\_to\_semantic\_id** (`list[dict[int, int]]` or `dict[int, int]`, *optional*) —
  A mapping between object instance ids and class ids.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>
* **do\_split\_image** (`bool`, *optional*, defaults to `False`) —
  Whether to split the input images into overlapping patches for semantic segmentation. If set to `True`, the
  input images will be split into patches of size `size["shortest_edge"]` with an overlap between patches.
  Otherwise, the input images will be padded to the target size.
* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
* **ignore\_index** (`int`, *optional*) —
  Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
  denoted with 0 (background) will be replaced with `ignore_index`.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt_fast.py#L393)

( outputs target\_sizes: list size: typing.Optional[dict[str, int]] = None  )

Post-processes model outputs into final semantic segmentation prediction.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt_fast.py#L482)

( outputs target\_sizes: list threshold: float = 0.8 size: typing.Optional[dict[str, int]] = None  )

Post-processes model outputs into Instance Segmentation Predictions.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/image_processing_eomt_fast.py#L425)

( outputs target\_sizes: list threshold: float = 0.8 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 stuff\_classes: typing.Optional[list[int]] = None size: typing.Optional[dict[str, int]] = None  )

Post-processes model outputs into final panoptic segmentation prediction.

## EomtConfig

### class transformers.EomtConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/configuration_eomt.py#L25)

( hidden\_size = 1024 num\_hidden\_layers = 24 num\_attention\_heads = 16 mlp\_ratio = 4 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-06 image\_size = 640 patch\_size = 16 num\_channels = 3 layerscale\_value = 1.0 drop\_path\_rate = 0.0 num\_upscale\_blocks = 2 attention\_dropout = 0.0 use\_swiglu\_ffn = False num\_blocks = 4 no\_object\_weight: float = 0.1 class\_weight: float = 2.0 mask\_weight: float = 5.0 dice\_weight: float = 5.0 train\_num\_points: int = 12544 oversample\_ratio: float = 3.0 importance\_sample\_ratio: float = 0.75 num\_queries = 200 num\_register\_tokens = 4 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the hidden representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads in each attention layer.
* **mlp\_ratio** (`int`, *optional*, defaults to 4) —
  Ratio of the MLP hidden dimensionality to the hidden size.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings and encoder.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`int`, *optional*, defaults to 640) —
  The size (resolution) of each input image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **layerscale\_value** (`float`, *optional*, defaults to 1.0) —
  Initial value for the LayerScale parameter.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  The stochastic depth rate (drop path) used during training.
* **num\_upscale\_blocks** (`int`, *optional*, defaults to 2) —
  Number of upsampling blocks used in the decoder or segmentation head.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  Dropout probability applied after attention projection.
* **use\_swiglu\_ffn** (`bool`, *optional*, defaults to `False`) —
  Whether to use the SwiGLU feedforward neural network.
* **num\_blocks** (`int`, *optional*, defaults to 4) —
  Number of feature blocks or stages in the architecture.
* **no\_object\_weight** (`float`, *optional*, defaults to 0.1) —
  Loss weight for the ‘no object’ class in panoptic/instance segmentation.
* **class\_weight** (`float`, *optional*, defaults to 2.0) —
  Loss weight for classification targets.
* **mask\_weight** (`float`, *optional*, defaults to 5.0) —
  Loss weight for mask prediction.
* **dice\_weight** (`float`, *optional*, defaults to 5.0) —
  Loss weight for the dice loss component.
* **train\_num\_points** (`int`, *optional*, defaults to 12544) —
  Number of points to sample for mask loss computation during training.
* **oversample\_ratio** (`float`, *optional*, defaults to 3.0) —
  Oversampling ratio used in point sampling for mask training.
* **importance\_sample\_ratio** (`float`, *optional*, defaults to 0.75) —
  Ratio of points to sample based on importance during training.
* **num\_queries** (`int`, *optional*, defaults to 200) —
  Number of object queries in the Transformer.
* **num\_register\_tokens** (`int`, *optional*, defaults to 4) —
  Number of learnable register tokens added to the transformer input.

This is the configuration class to store the configuration of a [EomtForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtForUniversalSegmentation). It is used to instantiate an EoMT model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the EoMT
[tue-mps/coco\_panoptic\_eomt\_large\_640](https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import EomtConfig, EomtForUniversalSegmentation

>>> # Initialize configuration
>>> config = EomtConfig()

>>> # Initialize model
>>> model = EomtForUniversalSegmentation(config)

>>> # Access config
>>> config = model.config
```

## EomtForUniversalSegmentation

### class transformers.EomtForUniversalSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/modeling_eomt.py#L1033)

( config: EomtConfig  )

Parameters

* **config** ([EomtConfig](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The EoMT Model with head on top for instance/semantic/panoptic segmentation.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/eomt/modeling_eomt.py#L1091)

( pixel\_values: Tensor mask\_labels: typing.Optional[list[torch.Tensor]] = None class\_labels: typing.Optional[list[torch.Tensor]] = None patch\_offsets: typing.Optional[list[torch.Tensor]] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.eomt.modeling_eomt.EomtForUniversalSegmentationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EomtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtImageProcessor). See [EomtImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [EomtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtImageProcessor) for processing images).
* **mask\_labels** (`list[torch.Tensor]`, *optional*) —
  list of mask labels of shape `(num_labels, height, width)` to be fed to a model
* **class\_labels** (`list[torch.LongTensor]`, *optional*) —
  list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
  labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
* **patch\_offsets** (`list[torch.Tensor]`, *optional*) —
  list of tuples indicating the image index and start and end positions of patches for semantic segementation.

Returns

`transformers.models.eomt.modeling_eomt.EomtForUniversalSegmentationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.eomt.modeling_eomt.EomtForUniversalSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EomtConfig](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtConfig)) and inputs.

* **loss** (`torch.Tensor`, *optional*) — The computed loss, returned when labels are present.
* **class\_queries\_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) — A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
  query. Note the `+ 1` is needed because we incorporate the null class.
* **masks\_queries\_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) — A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
  query.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Last hidden states (final feature map) of the last layer.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states all layers of the model.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Self and Cross Attentions weights from transformer decoder.
* **patch\_offsets** (`list[torch.Tensor]`, *optional*) — list of tuples indicating the image index and start and end positions of patches for semantic segementation.

The [EomtForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtForUniversalSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/eomt.md)
