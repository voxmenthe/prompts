*This model was released on 2022-09-22 and added to Hugging Face Transformers on 2022-12-16.*

# Swin2SR

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Swin2SR model was proposed in [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://huggingface.co/papers/2209.11345) by Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte.
Swin2SR improves the [SwinIR](https://github.com/JingyunLiang/SwinIR/) model by incorporating [Swin Transformer v2](swinv2) layers which mitigates issues such as training instability, resolution gaps between pre-training
and fine-tuning, and hunger on data.

The abstract from the paper is the following:

*Compression plays an important role on the efficient transmission and storage of images and videos through band-limited systems such as streaming services, virtual reality or videogames. However, compression unavoidably leads to artifacts and the loss of the original information, which may severely degrade the visual quality. For these reasons, quality enhancement of compressed images has become a popular research topic. While most state-of-the-art image restoration methods are based on convolutional neural networks, other transformers-based methods such as SwinIR, show impressive performance on these tasks.
In this paper, we explore the novel Swin Transformer V2, to improve SwinIR for image super-resolution, and in particular, the compressed input scenario. Using this method we can tackle the major issues in training transformer vision models, such as training instability, resolution gaps between pre-training and fine-tuning, and hunger on data. We conduct experiments on three representative tasks: JPEG compression artifacts removal, image super-resolution (classical and lightweight), and compressed image super-resolution. Experimental results demonstrate that our method, Swin2SR, can improve the training convergence and performance of SwinIR, and is a top-5 solution at the “AIM 2022 Challenge on Super-Resolution of Compressed Image and Video”.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/swin2sr_architecture.png) Swin2SR architecture. Taken from the [original paper.](https://huggingface.co/papers/2209.11345)

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/mv-lab/swin2sr).

## Resources

Demo notebooks for Swin2SR can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Swin2SR).

A demo Space for image super-resolution with SwinSR can be found [here](https://huggingface.co/spaces/jjourney1125/swin2sr).

## Swin2SRImageProcessor

### class transformers.Swin2SRImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/image_processing_swin2sr.py#L39)

( do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_pad: bool = True pad\_size: int = 8 \*\*kwargs  )

Parameters

* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.

Constructs a Swin2SR image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/image_processing_swin2sr.py#L110)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_pad: typing.Optional[bool] = None pad\_size: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image to make the height and width divisible by `window_size`.
* **pad\_size** (`int`, *optional*, defaults to 32) —
  The size of the sliding window for the local attention.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of typ, input\_data\_format=input\_data\_format
    `tf.Tensor`.
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

## Swin2SRImageProcessorFast

### class transformers.Swin2SRImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/image_processing_swin2sr_fast.py#L60)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.swin2sr.image\_processing\_swin2sr\_fast.Swin2SRFastImageProcessorKwargs]  )

Constructs a fast Swin2Sr image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/image_processing_swin2sr_fast.py#L70)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.swin2sr.image\_processing\_swin2sr\_fast.Swin2SRFastImageProcessorKwargs]  )

## Swin2SRConfig

### class transformers.Swin2SRConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/configuration_swin2sr.py#L24)

( image\_size = 64 patch\_size = 1 num\_channels = 3 num\_channels\_out = None embed\_dim = 180 depths = [6, 6, 6, 6, 6, 6] num\_heads = [6, 6, 6, 6, 6, 6] window\_size = 8 mlp\_ratio = 2.0 qkv\_bias = True hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 drop\_path\_rate = 0.1 hidden\_act = 'gelu' use\_absolute\_embeddings = False initializer\_range = 0.02 layer\_norm\_eps = 1e-05 upscale = 2 img\_range = 1.0 resi\_connection = '1conv' upsampler = 'pixelshuffle' \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 64) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 1) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_channels\_out** (`int`, *optional*, defaults to `num_channels`) —
  The number of output channels. If not set, it will be set to `num_channels`.
* **embed\_dim** (`int`, *optional*, defaults to 180) —
  Dimensionality of patch embedding.
* **depths** (`list(int)`, *optional*, defaults to `[6, 6, 6, 6, 6, 6]`) —
  Depth of each layer in the Transformer encoder.
* **num\_heads** (`list(int)`, *optional*, defaults to `[6, 6, 6, 6, 6, 6]`) —
  Number of attention heads in each layer of the Transformer encoder.
* **window\_size** (`int`, *optional*, defaults to 8) —
  Size of windows.
* **mlp\_ratio** (`float`, *optional*, defaults to 2.0) —
  Ratio of MLP hidden dimensionality to embedding dimensionality.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not a learnable bias should be added to the queries, keys and values.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings and encoder.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.1) —
  Stochastic depth rate.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
  `"selu"` and `"gelu_new"` are supported.
* **use\_absolute\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add absolute position embeddings to the patch embeddings.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **upscale** (`int`, *optional*, defaults to 2) —
  The upscale factor for the image. 2/3/4/8 for image super resolution, 1 for denoising and compress artifact
  reduction
* **img\_range** (`float`, *optional*, defaults to 1.0) —
  The range of the values of the input image.
* **resi\_connection** (`str`, *optional*, defaults to `"1conv"`) —
  The convolutional block to use before the residual connection in each stage.
* **upsampler** (`str`, *optional*, defaults to `"pixelshuffle"`) —
  The reconstruction reconstruction module. Can be ‘pixelshuffle’/‘pixelshuffledirect’/‘nearest+conv’/None.

This is the configuration class to store the configuration of a [Swin2SRModel](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRModel). It is used to instantiate a Swin
Transformer v2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Swin Transformer v2
[caidas/swin2sr-classicalsr-x2-64](https://huggingface.co/caidas/swin2sr-classicalsr-x2-64) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Swin2SRConfig, Swin2SRModel

>>> # Initializing a Swin2SR caidas/swin2sr-classicalsr-x2-64 style configuration
>>> configuration = Swin2SRConfig()

>>> # Initializing a model (with random weights) from the caidas/swin2sr-classicalsr-x2-64 style configuration
>>> model = Swin2SRModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Swin2SRModel

### class transformers.Swin2SRModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/modeling_swin2sr.py#L743)

( config  )

Parameters

* **config** ([Swin2SRModel](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Swin2Sr Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/modeling_swin2sr.py#L793)

( pixel\_values: FloatTensor head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Swin2SRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRImageProcessor). See [Swin2SRImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [Swin2SRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRImageProcessor) for processing images).
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

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Swin2SRConfig](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Swin2SRModel](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## Swin2SRForImageSuperResolution

### class transformers.Swin2SRForImageSuperResolution

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/modeling_swin2sr.py#L992)

( config  )

Parameters

* **config** ([Swin2SRForImageSuperResolution](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRForImageSuperResolution)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin2sr/modeling_swin2sr.py#L1019)

( pixel\_values: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.ImageSuperResolutionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Swin2SRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRImageProcessor). See [Swin2SRImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [Swin2SRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.ImageSuperResolutionOutput` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.ImageSuperResolutionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Swin2SRConfig](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Reconstruction loss.
* **reconstruction** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Reconstructed images, possibly upscaled.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Swin2SRForImageSuperResolution](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRForImageSuperResolution) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

>>> processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
>>> model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

>>> url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> # prepare image for the model
>>> inputs = processor(image, return_tensors="pt")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
>>> output = np.moveaxis(output, source=0, destination=-1)
>>> output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
>>> # you can visualize `output` with `Image.fromarray`
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/swin2sr.md)
