*This model was released on 2021-03-29 and added to Hugging Face Transformers on 2023-07-11.*

# Video Vision Transformer (ViViT)

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Vivit model was proposed in [ViViT: A Video Vision Transformer](https://huggingface.co/papers/2103.15691) by Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.
The paper proposes one of the first successful pure-transformer based set of models for video understanding.

The abstract from the paper is the following:

*We present pure-transformer based models for video classification, drawing upon the recent success of such models in image classification. Our model extracts spatio-temporal tokens from the input video, which are then encoded by a series of transformer layers. In order to handle the long sequences of tokens encountered in video, we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. Although transformer-based models are known to only be effective when large training datasets are available, we show how we can effectively regularise the model during training and leverage pretrained image models to be able to train on comparatively small datasets. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple video classification benchmarks including Kinetics 400 and 600, Epic Kitchens, Something-Something v2 and Moments in Time, outperforming prior methods based on deep 3D convolutional networks.*

This model was contributed by [jegormeister](https://huggingface.co/jegormeister). The original code (written in JAX) can be found [here](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit).

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.


```
from transformers import VivitModel
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `google/vivit-b-16x2-kinetics400` model, we saw the following speedups during inference.

### Training

| num\_training\_steps | batch\_size | is cuda | Speedup (%) | Eager peak mem (MB) | sdpa peak mem (MB) | Mem saving (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1 | True | 7.122 | 2575.28 | 5932.54 | 130.364 |

### Inference

| num\_batches | batch\_size | is cuda | is half | Speedup (%) | Mem eager (MB) | Mem BT (MB) | Mem saved (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 1 | True | False | 15.422 | 715.807 | 317.079 | 125.75 |
| 20 | 2 | True | False | 17.146 | 1234.75 | 447.175 | 176.122 |
| 20 | 4 | True | False | 18.093 | 2275.82 | 709.864 | 220.6 |
| 20 | 8 | True | False | 19.284 | 4358.19 | 1233.24 | 253.393 |

## VivitConfig

### class transformers.VivitConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/configuration_vivit.py#L24)

( image\_size = 224 num\_frames = 32 tubelet\_size = [2, 16, 16] num\_channels = 3 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu\_fast' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-06 qkv\_bias = True \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **num\_frames** (`int`, *optional*, defaults to 32) —
  The number of frames in each video.
* **tubelet\_size** (`list[int]`, *optional*, defaults to `[2, 16, 16]`) —
  The size (resolution) of each tubelet.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu_fast"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"`, `"gelu_fast"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.

This is the configuration class to store the configuration of a [VivitModel](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitModel). It is used to instantiate a ViViT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ViViT
[google/vivit-b-16x2-kinetics400](https://huggingface.co/google/vivit-b-16x2-kinetics400) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import VivitConfig, VivitModel

>>> # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
>>> configuration = VivitConfig()

>>> # Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
>>> model = VivitModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VivitImageProcessor

### class transformers.VivitImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/image_processing_vivit.py#L66)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00784313725490196 offset: bool = True do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 256}`):
  Size of the output image after resizing. The shortest edge of the image will be resized to
  `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overridden by
  `size` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
  parameter in the `preprocess` method.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
  `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/127.5`) —
  Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
  in the `preprocess` method.
* **offset** (`bool`, *optional*, defaults to `True`) —
  Whether to scale the image in both negative and positive directions. Can be overridden by the `offset` in
  the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a Vivit image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/image_processing_vivit.py#L289)

( videos: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None offset: typing.Optional[bool] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **videos** (`ImageInput`) —
  Video frames to preprocess. Expects a single or batch of video frames with pixel values ranging from 0
  to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after applying resize.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_centre_crop`) —
  Whether to centre crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the image after applying the centre crop.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between `[-1 - 1]` if `offset` is `True`, `[0, 1]` otherwise.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **offset** (`bool`, *optional*, defaults to `self.offset`) —
  Whether to scale the image in both negative and positive directions.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
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
  + Unset: Use the inferred channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## VivitModel

### class transformers.VivitModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/modeling_vivit.py#L422)

( config: VivitConfig add\_pooling\_layer: bool = True  )

Parameters

* **config** ([VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Vivit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/modeling_vivit.py#L454)

( pixel\_values: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None interpolate\_pos\_encoding: bool = False \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VivitModel](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import av
>>> import numpy as np

>>> from transformers import VivitImageProcessor, VivitModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)


>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`list[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices


>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> # sample 32 frames
>>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container=container, indices=indices)

>>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
>>> model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")

>>> # prepare video for the model
>>> inputs = image_processor(list(video), return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 3137, 768]
```

## VivitForVideoClassification

### class transformers.VivitForVideoClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/modeling_vivit.py#L566)

( config: VivitConfig  )

Parameters

* **config** ([VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViViT Transformer model with a video classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for Kinetics-400.

Note that it’s possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
position embeddings to the higher resolution.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vivit/modeling_vivit.py#L579)

( pixel\_values: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None interpolate\_pos\_encoding: bool = False \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VivitForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitForVideoClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import av
>>> import numpy as np
>>> import torch

>>> from transformers import VivitImageProcessor, VivitForVideoClassification
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)


>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`list[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices


>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> # sample 32 frames
>>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container=container, indices=indices)

>>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
>>> model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

>>> inputs = image_processor(list(video), return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
...     logits = outputs.logits

>>> # model predicts one of the 400 Kinetics-400 classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
LABEL_116
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vivit.md)
