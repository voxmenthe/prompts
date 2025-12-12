*This model was released on 2021-04-02 and added to Hugging Face Transformers on 2022-06-01.*

# LeViT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The LeViT model was proposed in [LeViT: Introducing Convolutions to Vision Transformers](https://huggingface.co/papers/2104.01136) by Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze. LeViT improves the [Vision Transformer (ViT)](vit) in performance and efficiency by a few architectural differences such as activation maps with decreasing resolutions in Transformers and the introduction of an attention bias to integrate positional information.

The abstract from the paper is the following:

*We design a family of image classification architectures that optimize the trade-off between accuracy
and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures,
which are competitive on highly parallel processing hardware. We revisit principles from the extensive
literature on convolutional neural networks to apply them to transformers, in particular activation maps
with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information
in vision transformers. As a result, we propose LeVIT: a hybrid neural network for fast inference image classification.
We consider different measures of efficiency on different hardware platforms, so as to best reflect a wide range of
application scenarios. Our extensive experiments empirically validate our technical choices and show they are suitable
to most architectures. Overall, LeViT significantly outperforms existing convnets and vision transformers with respect
to the speed/accuracy tradeoff. For example, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/levit_architecture.png) LeViT Architecture. Taken from the [original paper](https://huggingface.co/papers/2104.01136).

This model was contributed by [anugunj](https://huggingface.co/anugunj). The original code can be found [here](https://github.com/facebookresearch/LeViT).

## Usage tips

* Compared to ViT, LeViT models use an additional distillation head to effectively learn from a teacher (which, in the LeViT paper, is a ResNet like-model). The distillation head is learned through backpropagation under supervision of a ResNet like-model. They also draw inspiration from convolution neural networks to use activation maps with decreasing resolutions to increase the efficiency.
* There are 2 ways to fine-tune distilled models, either (1) in a classic way, by only placing a prediction head on top
  of the final hidden state and not using the distillation head, or (2) by placing both a prediction head and distillation
  head on top of the final hidden state. In that case, the prediction head is trained using regular cross-entropy between
  the prediction of the head and the ground-truth label, while the distillation prediction head is trained using hard distillation
  (cross-entropy between the prediction of the distillation head and the label predicted by the teacher). At inference time,
  one takes the average prediction between both heads as final prediction. (2) is also called “fine-tuning with distillation”,
  because one relies on a teacher that has already been fine-tuned on the downstream dataset. In terms of models, (1) corresponds
  to [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification) and (2) corresponds to [LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher).
* All released checkpoints were pre-trained and fine-tuned on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)
  (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes). only. No external data was used. This is in
  contrast with the original ViT model, which used external data like the JFT-300M dataset/Imagenet-21k for
  pre-training.
* The authors of LeViT released 5 trained LeViT models, which you can directly plug into [LevitModel](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitModel) or [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification).
  Techniques like data augmentation, optimization, and regularization were used in order to simulate training on a much larger dataset
  (while only using ImageNet-1k for pre-training). The 5 variants available are (all trained on images of size 224x224):
  *facebook/levit-128S*, *facebook/levit-128*, *facebook/levit-192*, *facebook/levit-256* and
  *facebook/levit-384*. Note that one should use [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor) in order to
  prepare images for the model.
* [LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher) currently supports only inference and not training or fine-tuning.
* You can check out demo notebooks regarding inference as well as fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)
  (you can just replace [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) by [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor) and [ViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForImageClassification) by [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification) or [LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher)).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LeViT.

Image Classification

* [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
* See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## LevitConfig

### class transformers.LevitConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/configuration_levit.py#L30)

( image\_size = 224 num\_channels = 3 kernel\_size = 3 stride = 2 padding = 1 patch\_size = 16 hidden\_sizes = [128, 256, 384] num\_attention\_heads = [4, 8, 12] depths = [4, 4, 4] key\_dim = [16, 16, 16] drop\_path\_rate = 0 mlp\_ratio = [2, 2, 2] attention\_ratio = [2, 2, 2] initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 224) —
  The size of the input image.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input image.
* **kernel\_size** (`int`, *optional*, defaults to 3) —
  The kernel size for the initial convolution layers of patch embedding.
* **stride** (`int`, *optional*, defaults to 2) —
  The stride size for the initial convolution layers of patch embedding.
* **padding** (`int`, *optional*, defaults to 1) —
  The padding size for the initial convolution layers of patch embedding.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The patch size for embeddings.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[128, 256, 384]`) —
  Dimension of each of the encoder blocks.
* **num\_attention\_heads** (`list[int]`, *optional*, defaults to `[4, 8, 12]`) —
  Number of attention heads for each attention layer in each block of the Transformer encoder.
* **depths** (`list[int]`, *optional*, defaults to `[4, 4, 4]`) —
  The number of layers in each encoder block.
* **key\_dim** (`list[int]`, *optional*, defaults to `[16, 16, 16]`) —
  The size of key in each of the encoder blocks.
* **drop\_path\_rate** (`int`, *optional*, defaults to 0) —
  The dropout probability for stochastic depths, used in the blocks of the Transformer encoder.
* **mlp\_ratios** (`list[int]`, *optional*, defaults to `[2, 2, 2]`) —
  Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
  encoder blocks.
* **attention\_ratios** (`list[int]`, *optional*, defaults to `[2, 2, 2]`) —
  Ratio of the size of the output dimension compared to input dimension of attention layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [LevitModel](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitModel). It is used to instantiate a LeViT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LeViT
[facebook/levit-128S](https://huggingface.co/facebook/levit-128S) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LevitConfig, LevitModel

>>> # Initializing a LeViT levit-128S style configuration
>>> configuration = LevitConfig()

>>> # Initializing a model (with random weights) from the levit-128S style configuration
>>> model = LevitModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LevitFeatureExtractor

### class transformers.LevitFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/feature_extraction_levit.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## LevitImageProcessor

### class transformers.LevitImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/image_processing_levit.py#L49)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, collections.abc.Iterable[float], NoneType] = [0.485, 0.456, 0.406] image\_std: typing.Union[float, collections.abc.Iterable[float], NoneType] = [0.229, 0.224, 0.225] \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Wwhether to resize the shortest edge of the input to int(256/224 \*`size`). Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]`, *optional*, defaults to `{"shortest_edge" -- 224}`):
  Size of the output image after resizing. If size is a dict with keys “width” and “height”, the image will
  be resized to `(size["height"], size["width"])`. If size is a dict with key “shortest\_edge”, the shortest
  edge value `c` is rescaled to `int(c * (256/224))`. The smaller edge of the image will be matched to this
  value i.e, if height > width, then image will be rescaled to `(size["shortest_egde"] * height / width, size["shortest_egde"])`. Can be overridden by the `size` parameter in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether or not to center crop the input to `(crop_size["height"], crop_size["width"])`. Can be overridden
  by the `do_center_crop` parameter in the `preprocess` method.
* **crop\_size** (`Dict`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Desired image size after `center_crop`. Can be overridden by the `crop_size` parameter in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
  `preprocess` method.
* **image\_mean** (`list[int]`, *optional*, defaults to `[0.485, 0.456, 0.406]`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`list[int]`, *optional*, defaults to `[0.229, 0.224, 0.225]`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a LeViT image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/image_processing_levit.py#L177)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, collections.abc.Iterable[float], NoneType] = None image\_std: typing.Union[float, collections.abc.Iterable[float], NoneType] = None return\_tensors: typing.Optional[transformers.utils.generic.TensorType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
  from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the output image after resizing. If size is a dict with keys “width” and “height”, the image
  will be resized to (height, width). If size is a dict with key “shortest\_edge”, the shortest edge value
  `c` is rescaled to int(`c`  *(256/224)). The smaller edge of the image will be matched to this value
  i.e, if height > width, then image will be rescaled to (size*  height / width, size).
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  Resampling filter to use when resiizing the image.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the output image after center cropping. Crops images to (crop\_size[“height”],
  crop\_size[“width”]).
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image pixel values by `rescaling_factor` - typical to values between 0 and 1.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Factor to rescale the image pixel values by.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image pixel values by `image_mean` and `image_std`.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to normalize the image pixel values by.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to normalize the image pixel values by.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`str` or `ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images to be used as input to a LeViT model.

## LevitImageProcessorFast

### class transformers.LevitImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/image_processing_levit_fast.py#L37)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Levit image processor.

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

## LevitModel

### class transformers.LevitModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/modeling_levit.py#L490)

( config  )

Parameters

* **config** ([LevitModel](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Levit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/modeling_levit.py#L499)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor). See [LevitImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LevitConfig](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [LevitModel](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## LevitForImageClassification

### class transformers.LevitForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/modeling_levit.py#L542)

( config  )

Parameters

* **config** ([LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/modeling_levit.py#L559)

( pixel\_values: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor). See [LevitImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LevitConfig](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, LevitForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/levit-128S")
>>> model = LevitForImageClassification.from_pretrained("facebook/levit-128S")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

## LevitForImageClassificationWithTeacher

### class transformers.LevitForImageClassificationWithTeacher

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/modeling_levit.py#L622)

( config  )

Parameters

* **config** ([LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and
a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::
This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
supported.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/levit/modeling_levit.py#L644)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.levit.modeling_levit.LevitForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor). See [LevitImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.levit.modeling_levit.LevitForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.levit.modeling_levit.LevitForImageClassificationWithTeacherOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LevitConfig](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitConfig)) and inputs.

* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Prediction scores as the average of the `cls_logits` and `distillation_logits`.
* **cls\_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
  class token).
* **distillation\_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
  distillation token).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, LevitForImageClassificationWithTeacher
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/levit-128S")
>>> model = LevitForImageClassificationWithTeacher.from_pretrained("facebook/levit-128S")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/levit.md)
