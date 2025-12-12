*This model was released on 2024-10-02 and added to Hugging Face Transformers on 2025-02-10.*

# DepthPro

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The DepthPro model was proposed in [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://huggingface.co/papers/2410.02073) by Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun.

DepthPro is a foundation model for zero-shot metric monocular depth estimation, designed to generate high-resolution depth maps with remarkable sharpness and fine-grained details. It employs a multi-scale Vision Transformer (ViT)-based architecture, where images are downsampled, divided into patches, and processed using a shared Dinov2 encoder. The extracted patch-level features are merged, upsampled, and refined using a DPT-like fusion stage, enabling precise depth estimation.

The abstract from the paper is the following:

*We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_pro_teaser.png) DepthPro Outputs. Taken from the [official code](https://github.com/apple/ml-depth-pro).

This model was contributed by [geetu040](https://github.com/geetu040). The original code can be found [here](https://github.com/apple/ml-depth-pro).

## Usage Tips

The DepthPro model processes an input image by first downsampling it at multiple scales and splitting each scaled version into patches. These patches are then encoded using a shared Vision Transformer (ViT)-based Dinov2 patch encoder, while the full image is processed by a separate image encoder. The extracted patch features are merged into feature maps, upsampled, and fused using a DPT-like decoder to generate the final depth estimation. If enabled, an additional Field of View (FOV) encoder processes the image for estimating the camera’s field of view, aiding in depth accuracy.


```
>>> import requests
>>> from PIL import Image
>>> import torch
>>> from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation, infer_device

>>> device = infer_device()

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
>>> model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

>>> inputs = image_processor(images=image, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs, target_sizes=[(image.height, image.width)],
... )

>>> field_of_view = post_processed_output[0]["field_of_view"]
>>> focal_length = post_processed_output[0]["focal_length"]
>>> depth = post_processed_output[0]["predicted_depth"]
>>> depth = (depth - depth.min()) / depth.max()
>>> depth = depth * 255.
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint8"))
```

### Architecture and Configuration

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_pro_architecture.png) DepthPro architecture. Taken from the [original paper](https://huggingface.co/papers/2410.02073).

The `DepthProForDepthEstimation` model uses a `DepthProEncoder`, for encoding the input image and a `FeatureFusionStage` for fusing the output features from encoder.

The `DepthProEncoder` further uses two encoders:

* `patch_encoder`
  + Input image is scaled with multiple ratios, as specified in the `scaled_images_ratios` configuration.
  + Each scaled image is split into smaller **patches** of size `patch_size` with overlapping areas determined by `scaled_images_overlap_ratios`.
  + These patches are processed by the **`patch_encoder`**
* `image_encoder`
  + Input image is also rescaled to `patch_size` and processed by the **`image_encoder`**

Both these encoders can be configured via `patch_model_config` and `image_model_config` respectively, both of which are separate `Dinov2Model` by default.

Outputs from both encoders (`last_hidden_state`) and selected intermediate states (`hidden_states`) from **`patch_encoder`** are fused by a `DPT`-based `FeatureFusionStage` for depth estimation.

### Field-of-View (FOV) Prediction

The network is supplemented with a focal length estimation head. A small convolutional head ingests frozen features from the depth estimation network and task-specific features from a separate ViT image encoder to predict the horizontal angular field-of-view.

The `use_fov_model` parameter in `DepthProConfig` controls whether **FOV prediction** is enabled. By default, it is set to `False` to conserve memory and computation. When enabled, the **FOV encoder** is instantiated based on the `fov_model_config` parameter, which defaults to a `Dinov2Model`. The `use_fov_model` parameter can also be passed when initializing the `DepthProForDepthEstimation` model.

The pretrained model at checkpoint `apple/DepthPro-hf` uses the FOV encoder. To use the pretrained-model without FOV encoder, set `use_fov_model=False` when loading the model, which saves computation.


```
>>> from transformers import DepthProForDepthEstimation
>>> model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", use_fov_model=False)
```

To instantiate a new model with FOV encoder, set `use_fov_model=True` in the config.


```
>>> from transformers import DepthProConfig, DepthProForDepthEstimation
>>> config = DepthProConfig(use_fov_model=True)
>>> model = DepthProForDepthEstimation(config)
```

Or set `use_fov_model=True` when initializing the model, which overrides the value in config.


```
>>> from transformers import DepthProConfig, DepthProForDepthEstimation
>>> config = DepthProConfig()
>>> model = DepthProForDepthEstimation(config, use_fov_model=True)
```

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.


```
from transformers import DepthProForDepthEstimation
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", attn_implementation="sdpa", dtype=torch.float16)
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `google/vit-base-patch16-224` model, we saw the following speedups during inference.

| Batch size | Average inference time (ms), eager mode | Average inference time (ms), sdpa model | Speed up, Sdpa / Eager (x) |
| --- | --- | --- | --- |
| 1 | 7 | 6 | 1.17 |
| 2 | 8 | 6 | 1.33 |
| 4 | 8 | 6 | 1.33 |
| 8 | 8 | 6 | 1.33 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DepthPro:

* Research Paper: [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://huggingface.co/papers/2410.02073)
* Official Implementation: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
* DepthPro Inference Notebook: [DepthPro Inference](https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/DepthPro_inference.ipynb)
* DepthPro for Super Resolution and Image Segmentation
  + Read blog on Medium: [Depth Pro: Beyond Depth](https://medium.com/@raoarmaghanshakir040/depth-pro-beyond-depth-9d822fc557ba)
  + Code on Github: [geetu040/depthpro-beyond-depth](https://github.com/geetu040/depthpro-beyond-depth)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DepthProConfig

### class transformers.DepthProConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/configuration_depth_pro.py#L27)

( fusion\_hidden\_size = 256 patch\_size = 384 initializer\_range = 0.02 intermediate\_hook\_ids = [11, 5] intermediate\_feature\_dims = [256, 256] scaled\_images\_ratios = [0.25, 0.5, 1] scaled\_images\_overlap\_ratios = [0.0, 0.5, 0.25] scaled\_images\_feature\_dims = [1024, 1024, 512] merge\_padding\_value = 3 use\_batch\_norm\_in\_fusion\_residual = False use\_bias\_in\_fusion\_residual = True use\_fov\_model = False num\_fov\_head\_layers = 2 image\_model\_config = None patch\_model\_config = None fov\_model\_config = None \*\*kwargs  )

Parameters

* **fusion\_hidden\_size** (`int`, *optional*, defaults to 256) —
  The number of channels before fusion.
* **patch\_size** (`int`, *optional*, defaults to 384) —
  The size (resolution) of each patch. This is also the image\_size for backbone model.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **intermediate\_hook\_ids** (`list[int]`, *optional*, defaults to `[11, 5]`) —
  Indices of the intermediate hidden states from the patch encoder to use for fusion.
* **intermediate\_feature\_dims** (`list[int]`, *optional*, defaults to `[256, 256]`) —
  Hidden state dimensions during upsampling for each intermediate hidden state in `intermediate_hook_ids`.
* **scaled\_images\_ratios** (`list[float]`, *optional*, defaults to `[0.25, 0.5, 1]`) —
  Ratios of scaled images to be used by the patch encoder.
* **scaled\_images\_overlap\_ratios** (`list[float]`, *optional*, defaults to `[0.0, 0.5, 0.25]`) —
  Overlap ratios between patches for each scaled image in `scaled_images_ratios`.
* **scaled\_images\_feature\_dims** (`list[int]`, *optional*, defaults to `[1024, 1024, 512]`) —
  Hidden state dimensions during upsampling for each scaled image in `scaled_images_ratios`.
* **merge\_padding\_value** (`int`, *optional*, defaults to 3) —
  When merging smaller patches back to the image size, overlapping sections of this size are removed.
* **use\_batch\_norm\_in\_fusion\_residual** (`bool`, *optional*, defaults to `False`) —
  Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
* **use\_bias\_in\_fusion\_residual** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the pre-activate residual units of the fusion blocks.
* **use\_fov\_model** (`bool`, *optional*, defaults to `False`) —
  Whether to use `DepthProFovModel` to generate the field of view.
* **num\_fov\_head\_layers** (`int`, *optional*, defaults to 2) —
  Number of convolution layers in the head of `DepthProFovModel`.
* **image\_model\_config** (`Union[dict[str, Any], PretrainedConfig]`, *optional*) —
  The configuration of the image encoder model, which is loaded using the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) API.
  By default, Dinov2 model is used as backbone.
* **patch\_model\_config** (`Union[dict[str, Any], PretrainedConfig]`, *optional*) —
  The configuration of the patch encoder model, which is loaded using the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) API.
  By default, Dinov2 model is used as backbone.
* **fov\_model\_config** (`Union[dict[str, Any], PretrainedConfig]`, *optional*) —
  The configuration of the fov encoder model, which is loaded using the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) API.
  By default, Dinov2 model is used as backbone.

This is the configuration class to store the configuration of a [DepthProModel](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProModel). It is used to instantiate a
DepthPro model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the DepthPro
[apple/DepthPro](https://huggingface.co/apple/DepthPro) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DepthProConfig, DepthProModel

>>> # Initializing a DepthPro apple/DepthPro style configuration
>>> configuration = DepthProConfig()

>>> # Initializing a model (with random weights) from the apple/DepthPro style configuration
>>> model = DepthProModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DepthProImageProcessor

### class transformers.DepthProImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/image_processing_depth_pro.py#L56)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"height" -- 1536, "width": 1536}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a DepthPro image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/image_processing_depth_pro.py#L191)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: typing.Optional[PIL.Image.Resampling] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
  resizing.
* **resample** (`PILImageResampling` filter, *optional*, defaults to `self.resample`) —
  `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
  an effect if `do_resize` is set to `True`.
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
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

#### post\_process\_depth\_estimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/image_processing_depth_pro.py#L317)

( outputs: DepthProDepthEstimatorOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple[int, int]], NoneType] = None  ) → `list[dict[str, TensorType]]`

Parameters

* **outputs** (`DepthProDepthEstimatorOutput`) —
  Raw outputs of the model.
* **target\_sizes** (`Optional[Union[TensorType, list[tuple[int, int]], None]]`, *optional*, defaults to `None`) —
  Target sizes to resize the depth predictions. Can be a tensor of shape `(batch_size, 2)`
  or a list of tuples `(height, width)` for each image in the batch. If `None`, no resizing
  is performed.

Returns

`list[dict[str, TensorType]]`

A list of dictionaries of tensors representing the processed depth
predictions, and field of view (degrees) and focal length (pixels) if `field_of_view` is given in `outputs`.

Raises

`ValueError`

* `ValueError` —
  If the lengths of `predicted_depths`, `fovs`, or `target_sizes` are mismatched.

Post-processes the raw depth predictions from the model to generate
final depth predictions which is caliberated using the field of view if provided
and resized to specified target sizes if provided.

## DepthProImageProcessorFast

### class transformers.DepthProImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/image_processing_depth_pro_fast.py#L55)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Depth Pro image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L639)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*args \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### post\_process\_depth\_estimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/image_processing_depth_pro_fast.py#L105)

( outputs: DepthProDepthEstimatorOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple[int, int]], NoneType] = None  ) → `list[dict[str, TensorType]]`

Parameters

* **outputs** (`DepthProDepthEstimatorOutput`) —
  Raw outputs of the model.
* **target\_sizes** (`Optional[Union[TensorType, list[tuple[int, int]], None]]`, *optional*, defaults to `None`) —
  Target sizes to resize the depth predictions. Can be a tensor of shape `(batch_size, 2)`
  or a list of tuples `(height, width)` for each image in the batch. If `None`, no resizing
  is performed.

Returns

`list[dict[str, TensorType]]`

A list of dictionaries of tensors representing the processed depth
predictions, and field of view (degrees) and focal length (pixels) if `field_of_view` is given in `outputs`.

Raises

`ValueError`

* `ValueError` —
  If the lengths of `predicted_depths`, `fovs`, or `target_sizes` are mismatched.

Post-processes the raw depth predictions from the model to generate
final depth predictions which is caliberated using the field of view if provided
and resized to specified target sizes if provided.

## DepthProModel

### class transformers.DepthProModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/modeling_depth_pro.py#L636)

( config  )

Parameters

* **config** ([DepthProModel](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Depth Pro Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/modeling_depth_pro.py#L648)

( pixel\_values: FloatTensor head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.depth_pro.modeling_depth_pro.DepthProOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DepthProImageProcessor](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProImageProcessor). See [DepthProImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DepthProImageProcessor](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.depth_pro.modeling_depth_pro.DepthProOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.depth_pro.modeling_depth_pro.DepthProOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DepthProConfig](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **features** (`Union[torch.FloatTensor, List[torch.FloatTensor]]`, *optional*) — Features from encoders. Can be a single feature or a list of features.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DepthProModel](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, DepthProModel

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> checkpoint = "apple/DepthPro-hf"
>>> processor = AutoProcessor.from_pretrained(checkpoint)
>>> model = DepthProModel.from_pretrained(checkpoint)

>>> # prepare image for the model
>>> inputs = processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     output = model(**inputs)

>>> output.last_hidden_state.shape
torch.Size([1, 35, 577, 1024])
```

## DepthProForDepthEstimation

### class transformers.DepthProForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/modeling_depth_pro.py#L1009)

( config use\_fov\_model = None  )

Parameters

* **config** ([DepthProForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProForDepthEstimation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **use\_fov\_model** (`bool`, *optional*) —
  Whether to use the field of view model.

DepthPro Model with a depth estimation head on top (consisting of 3 convolutional layers).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_pro/modeling_depth_pro.py#L1034)

( pixel\_values: FloatTensor head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.depth_pro.modeling_depth_pro.DepthProDepthEstimatorOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DepthProImageProcessor](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProImageProcessor). See [DepthProImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DepthProImageProcessor](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth depth estimation maps for computing the loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.depth_pro.modeling_depth_pro.DepthProDepthEstimatorOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.depth_pro.modeling_depth_pro.DepthProDepthEstimatorOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DepthProConfig](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **predicted\_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*, defaults to `None`) — Predicted depth for each pixel.
* **field\_of\_view** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned when `use_fov_model` is provided) — Field of View Scaler.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DepthProForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DepthProForDepthEstimation
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> checkpoint = "apple/DepthPro-hf"
>>> processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = DepthProForDepthEstimation.from_pretrained(checkpoint)

>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> model.to(device)

>>> # prepare image for the model
>>> inputs = processor(images=image, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = processor.post_process_depth_estimation(
...     outputs, target_sizes=[(image.height, image.width)],
... )

>>> # get the field of view (fov) predictions
>>> field_of_view = post_processed_output[0]["field_of_view"]
>>> focal_length = post_processed_output[0]["focal_length"]

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 255 / predicted_depth.max()
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint8"))
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/depth_pro.md)
