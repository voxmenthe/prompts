*This model was released on 2021-05-31 and added to Hugging Face Transformers on 2021-10-28.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# SegFormer

[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://huggingface.co/papers/2105.15203) is a semantic segmentation model that combines a hierarchical Transformer encoder (Mix Transformer, MiT) with a lightweight all-MLP decoder. It avoids positional encodings and complex decoders and achieves state-of-the-art performance on benchmarks like ADE20K and Cityscapes. This simple and lightweight design is more efficient and scalable.

The figure below illustrates the architecture of SegFormer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png)

You can find all the original SegFormer checkpoints under the [NVIDIA](https://huggingface.co/nvidia/models?search=segformer) organization.

This model was contributed by [nielsr](https://huggingface.co/nielsr).

Click on the SegFormer models in the right sidebar for more examples of how to apply SegFormer to different vision tasks.

The example below demonstrates semantic segmentation with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", torch_dtype=torch.float16)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

## Notes

* SegFormer works with **any input size**, padding inputs to be divisible by `config.patch_sizes`.
* The most important preprocessing step is to randomly crop and pad all images to the same size (such as 512x512 or 640x640) and normalize afterwards.
* Some datasets (ADE20k) uses the `0` index in the annotated segmentation as the background, but doesn’t include the “background” class in its labels. The `do_reduce_labels` argument in `SegformerForImageProcessor` is used to reduce all labels by `1`. To make sure no loss is computed for the background class, it replaces `0` in the annotated maps by `255`, which is the `ignore_index` of the loss function.

  Other datasets may include a background class and label though, in which case, `do_reduce_labels` should be `False`.


```
from transformers import SegformerImageProcessor
processor = SegformerImageProcessor(do_reduce_labels=True)
```

## Resources

* [Original SegFormer code (NVlabs)](https://github.com/NVlabs/SegFormer)
* [Fine-tuning blog post](https://huggingface.co/blog/fine-tune-segformer)
* [Tutorial notebooks (Niels Rogge)](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer)
* [Hugging Face demo space](https://huggingface.co/spaces/chansung/segformer-tf-transformers)

## SegformerConfig

### class transformers.SegformerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/configuration_segformer.py#L31)

( num\_channels = 3 num\_encoder\_blocks = 4 depths = [2, 2, 2, 2] sr\_ratios = [8, 4, 2, 1] hidden\_sizes = [32, 64, 160, 256] patch\_sizes = [7, 3, 3, 3] strides = [4, 2, 2, 2] num\_attention\_heads = [1, 2, 5, 8] mlp\_ratios = [4, 4, 4, 4] hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 classifier\_dropout\_prob = 0.1 initializer\_range = 0.02 drop\_path\_rate = 0.1 layer\_norm\_eps = 1e-06 decoder\_hidden\_size = 256 semantic\_loss\_ignore\_index = 255 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_encoder\_blocks** (`int`, *optional*, defaults to 4) —
  The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
* **depths** (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`) —
  The number of layers in each encoder block.
* **sr\_ratios** (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`) —
  Sequence reduction ratios in each encoder block.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[32, 64, 160, 256]`) —
  Dimension of each of the encoder blocks.
* **patch\_sizes** (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`) —
  Patch size before each encoder block.
* **strides** (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`) —
  Stride before each encoder block.
* **num\_attention\_heads** (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`) —
  Number of attention heads for each attention layer in each block of the Transformer encoder.
* **mlp\_ratios** (`list[int]`, *optional*, defaults to `[4, 4, 4, 4]`) —
  Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
  encoder blocks.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **classifier\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability before the classification head.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **decoder\_hidden\_size** (`int`, *optional*, defaults to 256) —
  The dimension of the all-MLP decode head.
* **semantic\_loss\_ignore\_index** (`int`, *optional*, defaults to 255) —
  The index that is ignored by the loss function of the semantic segmentation model.

This is the configuration class to store the configuration of a [SegformerModel](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerModel). It is used to instantiate an
SegFormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SegFormer
[nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SegformerModel, SegformerConfig

>>> # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
>>> configuration = SegformerConfig()

>>> # Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration
>>> model = SegformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SegformerFeatureExtractor

### class transformers.SegformerFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/feature_extraction_segformer.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer.py#L286)

( images segmentation\_maps = None \*\*kwargs  )

Preprocesses a batch of images and optionally segmentation maps.

Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
passed in as positional arguments.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer.py#L427)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## SegformerImageProcessor

### class transformers.SegformerImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer.py#L58)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_reduce\_labels: bool = False \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 512, "width": 512}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `False`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
  used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
  background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the
  `preprocess` method.

Constructs a Segformer image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer.py#L295)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_reduce\_labels: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  Segmentation map to preprocess.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after `resize` is applied.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer.py#L427)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## SegformerImageProcessorFast

### class transformers.SegformerImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer_fast.py#L71)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.segformer.image\_processing\_segformer\_fast.SegformerFastImageProcessorKwargs]  )

Constructs a fast Segformer image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer_fast.py#L99)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.segformer.image\_processing\_segformer\_fast.SegformerFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The segmentation maps to preprocess.
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
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/image_processing_segformer_fast.py#L204)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## SegformerModel

### class transformers.SegformerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L460)

( config  )

Parameters

* **config** ([SegformerModel](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Segformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L479)

( pixel\_values: FloatTensor output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor). See [SegformerImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerFeatureExtractor.__call__) for details (`processor_class` uses
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [SegformerModel](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## SegformerDecodeHead

### class transformers.SegformerDecodeHead

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L617)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L642)

( encoder\_hidden\_states: FloatTensor  )

## SegformerForImageClassification

### class transformers.SegformerForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L517)

( config  )

Parameters

* **config** ([SegformerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
states) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L530)

( pixel\_values: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.segformer.modeling_segformer.SegFormerImageClassifierOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor). See [SegformerImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerFeatureExtractor.__call__) for details (`processor_class` uses
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.segformer.modeling_segformer.SegFormerImageClassifierOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.segformer.modeling_segformer.SegFormerImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [SegformerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, SegformerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
>>> model = SegformerForImageClassification.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

## SegformerForSemanticSegmentation

### class transformers.SegformerForSemanticSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L680)

( config  )

Parameters

* **config** ([SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/segformer/modeling_segformer.py#L689)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor). See [SegformerImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerFeatureExtractor.__call__) for details (`processor_class` uses
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) — Classification scores for each pixel.

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
>>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
>>> list(logits.shape)
[1, 150, 128, 128]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/segformer.md)
