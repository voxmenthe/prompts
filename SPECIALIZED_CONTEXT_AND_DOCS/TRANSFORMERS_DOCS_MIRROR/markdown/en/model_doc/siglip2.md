*This model was released on 2025-02-20 and added to Hugging Face Transformers on 2025-02-21.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# SigLIP2

## Overview

[SigLIP2](https://huggingface.co/papers/2502.14786) is a family of multilingual vision-language encoders that builds on the [SigLIP](./siglip) training recipe. It includes decoder-based pretraining, self-distillation, and masked prediction to improve dense prediction tasks (segmentation, depth estimation, etc.). This model is available in two variants:

* NaFlex supports different resolutions and maintains the native image aspect ratio
* FixRes supports fixed resolutions and is backwards compatible with [SigLIP](./siglip)

You can find all the original SigLIP2 checkpoints under the [SigLIP2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) collection.

Click on the SigLIP2 models in the right sidebar for more examples of how to apply SigLIP2 to different image and text tasks.

The example below demonstrates zero-shot classification with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel (FixRes)

AutoModel (NaFlex)


```
import torch
from transformers import pipeline

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

pipeline = pipeline(task="zero-shot-image-classification", model="google/siglip2-base-patch16-224", device=0, dtype=torch.bfloat16)
pipeline(image, candidate_labels=candidate_labels)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.


```
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModel.from_pretrained("google/siglip2-large-patch16-512", quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

# follows the pipeline prompt template to get same results
texts = [f'This is a photo of {label}.' for label in candidate_labels]

# IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

## Notes

* Training is supported for DDP and FSDP on single-node multi-accelerator setups. However, it does not use [torch.distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) utilities which may limit the scalability of batch size.
* When using the standalone [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) make sure to pass `padding="max_length"` and `max_length=64` as that’s how the model was trained.
* Model was trained with *lowercased* text, so make sure your text labels are preprocessed the same way.
* To get the same results as the [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), a prompt template of `"This is a photo of {label}."` should be passed to the processor.
* The NaFlex variant processes different types of images at the appropriate resolution (using a larger resolution to process document images for example), while also minimizing the impact of aspect ratio distortion for certain inference tasks like OCR.

  NaFlex resizes the input image so the height and width are multiples of the patch size after resizing. It keeps the aspect ratio distortion as low as possible and produces a sequence length of at most the desired target sequence length (`max_num_patches`). After resizing, the image is split into a sequence of patches and a mask with padding information is added.
* Toggle the `attn_implementation` parameter to either `"sdpa"` or `"flash_attention_2"` to use a more memory-efficient attention.


  ```
  # pip install -U flash-attn --no-build-isolation

  from transformers import SiglipModel

  model = SiglipModel.from_pretrained(
      "google/siglip2-so400m-patch14-384",
      attn_implementation="flash_attention_2",
      dtype=torch.float16,
      device_map=device,
  )
  ```

## Siglip2Config

### class transformers.Siglip2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/configuration_siglip2.py#L203)

( text\_config = None vision\_config = None \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Siglip2TextConfig](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Siglip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionConfig).
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config) is the configuration class to store the configuration of a [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model). It is used to
instantiate a Siglip2 model according to the specified arguments, defining the text model and vision model configs.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Siglip2
[google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Siglip2Config, Siglip2Model

>>> # Initializing a Siglip2Config with google/siglip2-base-patch16-224 style configuration
>>> configuration = Siglip2Config()

>>> # Initializing a Siglip2Model (with random weights) from the google/siglip2-base-patch16-224 style configuration
>>> model = Siglip2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a Siglip2Config from a Siglip2TextConfig and a Siglip2VisionConfig
>>> from transformers import Siglip2TextConfig, Siglip2VisionConfig

>>> # Initializing a Siglip2Text and Siglip2Vision configuration
>>> config_text = Siglip2TextConfig()
>>> config_vision = Siglip2VisionConfig()

>>> config = Siglip2Config.from_text_vision_configs(config_text, config_vision)
```

## Siglip2TextConfig

### class transformers.Siglip2TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/configuration_siglip2.py#L29)

( vocab\_size = 32000 hidden\_size = 768 intermediate\_size = 3072 num\_hidden\_layers = 12 num\_attention\_heads = 12 max\_position\_embeddings = 64 hidden\_act = 'gelu\_pytorch\_tanh' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 pad\_token\_id = 1 bos\_token\_id = 49406 eos\_token\_id = 49407 projection\_size = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the Siglip2 text model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 64) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **pad\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the padding token in the vocabulary.
* **bos\_token\_id** (`int`, *optional*, defaults to 49406) —
  The id of the beginning-of-sequence token in the vocabulary.
* **eos\_token\_id** (`int`, *optional*, defaults to 49407) —
  The id of the end-of-sequence token in the vocabulary.
* **projection\_size** (`int`, *optional*, defaults to `hidden_size`) —
  The size of the projection head.

This is the configuration class to store the configuration of a [Siglip2TextModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextModel). It is used to instantiate a
Siglip2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the text encoder of the Siglip2
[google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Siglip2TextConfig, Siglip2TextModel

>>> # Initializing a Siglip2TextConfig with google/siglip2-base-patch16-224 style configuration
>>> configuration = Siglip2TextConfig()

>>> # Initializing a Siglip2TextModel (with random weights) from the google/siglip2-base-patch16-224 style configuration
>>> model = Siglip2TextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Siglip2VisionConfig

### class transformers.Siglip2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/configuration_siglip2.py#L121)

( hidden\_size = 768 intermediate\_size = 3072 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 num\_patches = 256 patch\_size = 16 hidden\_act = 'gelu\_pytorch\_tanh' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input images.
* **num\_patches** (`int`, *optional*, defaults to 256) —
  The number of patches in the image with the size of (`patch_size`, `patch_size`).
  The image is resized to fill maximum of this number of patches, and to preserve
  the aspect ratio. In case the resulted number of patches is lower, the image is
  padded in “patch” dimension.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.

This is the configuration class to store the configuration of a [Siglip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionModel). It is used to instantiate a
Siglip2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip2
[google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Siglip2VisionConfig, Siglip2VisionModel

>>> # Initializing a Siglip2VisionConfig with google/siglip2-base-patch16-naflex style configuration
>>> configuration = Siglip2VisionConfig()

>>> # Initializing a Siglip2VisionModel (with random weights) from the google/siglip2-base-patch16-naflex style configuration
>>> model = Siglip2VisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Siglip2ImageProcessor

### class transformers.Siglip2ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/image_processing_siglip2.py#L126)

( do\_resize: bool = True resample: PILImageResampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None patch\_size: int = 16 max\_num\_patches: int = 256 \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s dimensions to fit `max_num_patches` according to given `patch_size`.
  Can be overridden by `do_resize` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image by the specified mean and standard deviation. Can be overridden by
  `do_normalize` in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch the image will be split to.
* **max\_num\_patches** (`int`, *optional*, defaults to 256) —
  The image will be resized to have at most this number of patches,
  and then padded in “patch” dimension to match this number exactly.

Constructs a SigLIP2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/image_processing_siglip2.py#L193)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None resample: typing.Optional[ForwardRef('PILImageResampling')] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None patch\_size: typing.Optional[int] = None max\_num\_patches: typing.Optional[int] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **patch\_size** (`int`, *optional*, defaults to `self.patch_size`) —
  Patch size for processing, same as the patch size used in the model.
* **max\_num\_patches** (`int`, *optional*, defaults to `self.max_num_patches`) —
  Maximum number of patches per image, the image will be resized to have at most this number of patches.

Preprocess an image or batch of images.

## Siglip2ImageProcessorFast

### class transformers.Siglip2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/image_processing_siglip2_fast.py#L100)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.siglip2.image\_processing\_siglip2\_fast.Siglip2FastImageProcessorKwargs]  )

Constructs a fast Siglip2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/image_processing_siglip2_fast.py#L120)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.siglip2.image\_processing\_siglip2\_fast.Siglip2FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch the image will be split to.
* **max\_num\_patches** (`int`, *optional*, defaults to 256) —
  The image will be resized to have at most this number of patches,
  and then padded in “patch” dimension to match this number exactly.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Siglip2Processor

### class transformers.Siglip2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/processing_siglip2.py#L48)

( image\_processor tokenizer  )

Parameters

* **image\_processor** ([Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor)) —
  The image processor is a required input.
* **tokenizer** ([GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast)) —
  The tokenizer is a required input.

Constructs a Siglip2 processor which wraps a Siglip2 image processor and a Gemma tokenizer into a single processor.

[Siglip2Processor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Processor) offers all the functionalities of [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor) and [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Siglip2Model

### class transformers.Siglip2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L921)

( config: Siglip2Config  )

Parameters

* **config** ([Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Siglip2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L1059)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_attention\_mask: typing.Optional[torch.Tensor] = None spatial\_shapes: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.models.siglip2.modeling_siglip2.Siglip2Output` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor). See [Siglip2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Siglip2Processor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Processor) uses
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **spatial\_shapes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  Tensor containing the spatial dimensions (height, width) of the input images.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **return\_loss** (`bool`, *optional*) —
  Whether or not to return the contrastive loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.siglip2.modeling_siglip2.Siglip2Output` or `tuple(torch.FloatTensor)`

A `transformers.models.siglip2.modeling_siglip2.Siglip2Output` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [Siglip2TextModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [Siglip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [Siglip2TextModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [Siglip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionModel).

The [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
>>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
>>> # important: we pass `padding=max_length` since the model was trained with this
>>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image
>>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
>>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
31.9% that image 0 is 'a photo of 2 cats'
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L956)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [Siglip2TextModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextModel).

Examples:


```
>>> from transformers import AutoTokenizer, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
>>> tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

>>> # important: make sure to set padding="max_length" as that's how the model was trained
>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
>>> with torch.no_grad():
...     text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L1002)

( pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_attention\_mask: typing.Optional[torch.Tensor] = None spatial\_shapes: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor). See [Siglip2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Siglip2Processor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Processor) uses
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **spatial\_shapes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  Tensor containing the spatial dimensions (height, width) of the input images.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by
applying the projection layer to the pooled output of [Siglip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
>>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     image_features = model.get_image_features(**inputs)
```

## Siglip2TextModel

### class transformers.Siglip2TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L774)

( config: Siglip2TextConfig  )

Parameters

* **config** ([Siglip2TextConfig](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text model from Siglip2 without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L789)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config)) and inputs.

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

The [Siglip2TextModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, Siglip2TextModel

>>> model = Siglip2TextModel.from_pretrained("google/siglip2-base-patch16-224")
>>> tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

>>> # important: make sure to set padding="max_length" as that's how the model was trained
>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

## Siglip2VisionModel

### class transformers.Siglip2VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L861)

( config: Siglip2VisionConfig  )

Parameters

* **config** ([Siglip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from Siglip2 without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L876)

( pixel\_values: FloatTensor pixel\_attention\_mask: Tensor spatial\_shapes: LongTensor output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor). See [Siglip2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Siglip2Processor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Processor) uses
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **spatial\_shapes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  Tensor containing the spatial dimensions (height, width) of the input images.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config)) and inputs.

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

The [Siglip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Siglip2VisionModel

>>> model = Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-224")
>>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled features
```

## Siglip2ForImageClassification

### class transformers.Siglip2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L1171)

( config: Siglip2Config  )

Parameters

* **config** ([Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Siglip2 vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of
the patch tokens) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/siglip2/modeling_siglip2.py#L1192)

( pixel\_values: typing.Optional[torch.Tensor] = None pixel\_attention\_mask: typing.Optional[torch.Tensor] = None spatial\_shapes: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor). See [Siglip2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Siglip2Processor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Processor) uses
  [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **spatial\_shapes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  Tensor containing the spatial dimensions (height, width) of the input images.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Siglip2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, Siglip2ForImageClassification
>>> import torch
>>> from PIL import Image
>>> import requests

>>> torch.manual_seed(3)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # note: we are loading a `Siglip2Model` from the hub here,
>>> # so the head will be randomly initialized, hence the predictions will be random if seed is not set above.
>>> image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
>>> model = Siglip2ForImageClassification.from_pretrained("google/siglip2-base-patch16-224")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> # model predicts one of the two classes
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: LABEL_1
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)
