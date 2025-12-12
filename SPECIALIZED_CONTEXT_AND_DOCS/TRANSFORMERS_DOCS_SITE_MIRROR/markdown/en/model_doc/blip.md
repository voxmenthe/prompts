# BLIP

[BLIP](https://huggingface.co/papers/2201.12086) (Bootstrapped Language-Image Pretraining) is a vision-language pretraining (VLP) framework designed for *both* understanding and generation tasks. Most existing pretrained models are only good at one or the other. It uses a captioner to generate captions and a filter to remove the noisy captions. This increases training data quality and more effectively uses the messy web data.

You can find all the original BLIP checkpoints under the [BLIP](https://huggingface.co/collections/Salesforce/blip-models-65242f40f1491fbf6a9e9472) collection.

> [!TIP]
> This model was contributed by [ybelkada](https://huggingface.co/ybelkada).
>
> Click on the BLIP models in the right sidebar for more examples of how to apply BLIP to different vision language tasks.

The example below demonstrates how to visual question answering with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="visual-question-answering",
    model="Salesforce/blip-vqa-base",
    dtype=torch.float16,
    device=0
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
pipeline(question="What is the weather in this image?", image=url)
```

```python
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = AutoModelForVisualQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base",
    dtype=torch.float16,
    device_map="auto"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

question = "What is the weather in this image?"
inputs = processor(images=image, text=question, return_tensors="pt").to(model.device, torch.float16)

output = model.generate(**inputs)
processor.batch_decode(output, skip_special_tokens=True)[0]
```

## Resources

Refer to this [notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) to learn how to fine-tune BLIP for image captioning on a custom dataset.

## BlipConfig[[transformers.BlipConfig]]

#### transformers.BlipConfig[[transformers.BlipConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/configuration_blip.py#L230)

[BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig) is the configuration class to store the configuration of a [BlipModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipModel). It is used to instantiate
a BLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
a configuration with the defaults will yield a similar configuration to that of the BLIP-base
[Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import BlipConfig, BlipModel

>>> # Initializing a BlipConfig with Salesforce/blip-vqa-base style configuration
>>> configuration = BlipConfig()

>>> # Initializing a BlipPModel (with random weights) from the Salesforce/blip-vqa-base style configuration
>>> model = BlipModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a BlipConfig from a BlipTextConfig and a BlipVisionConfig

>>> # Initializing a BLIPText and BLIPVision configuration
>>> config_text = BlipTextConfig()
>>> config_vision = BlipVisionConfig()

>>> config = BlipConfig(text_config=config_text, vision_config=config_vision)
```

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [BlipTextConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextConfig).

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [BlipVisionConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionConfig).

projection_dim (`int`, *optional*, defaults to 512) : Dimensionality of text and vision projection layers.

logit_scale_init_value (`float`, *optional*, defaults to 2.6592) : The initial value of the *logit_scale* parameter. Default is used as per the original BLIP implementation.

image_text_hidden_size (`int`, *optional*, defaults to 256) : Dimensionality of the hidden state of the image-text fusion layer.

label_smoothing (float, optional, *optional*, defaults to 0.0) : A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in `Rethinking the Inception Architecture for Computer Vision `__. Default: :math:`0.0`.

kwargs (*optional*) : Dictionary of keyword arguments.

## BlipTextConfig[[transformers.BlipTextConfig]]

#### transformers.BlipTextConfig[[transformers.BlipTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/configuration_blip.py#L24)

This is the configuration class to store the configuration of a [BlipTextModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextModel). It is used to instantiate a BLIP
text model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the `BlipText` used by the [base
architectures](https://huggingface.co/Salesforce/blip-vqa-base).

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import BlipTextConfig, BlipTextModel

>>> # Initializing a BlipTextConfig with Salesforce/blip-vqa-base style configuration
>>> configuration = BlipTextConfig()

>>> # Initializing a BlipTextModel (with random weights) from the Salesforce/blip-vqa-base style configuration
>>> model = BlipTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30524) : Vocabulary size of the `Blip` text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [BlipModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipModel).

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

encoder_hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers from the vision model.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Transformer encoder.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

bos_token_id (`int`, *optional*, defaults to 30522) : The id of the `beginning-of-sequence` token.

eos_token_id (`int`, *optional*, defaults to 2) : The id of the `end-of-sequence` token.

pad_token_id (`int`, *optional*, defaults to 0) : The id of the `padding` token.

sep_token_id (`int`, *optional*, defaults to 102) : The id of the `separator` token.

is_decoder (`bool`, *optional*, defaults to `True`) : Whether the model is used as a decoder.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models).

label_smoothing (float, *optional*) : A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in `Rethinking the Inception Architecture for Computer Vision `__. Default: :math:`0.0`.

## BlipVisionConfig[[transformers.BlipVisionConfig]]

#### transformers.BlipVisionConfig[[transformers.BlipVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/configuration_blip.py#L148)

This is the configuration class to store the configuration of a [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel). It is used to instantiate a
BLIP vision model according to the specified arguments, defining the model architecture. Instantiating a
configuration defaults will yield a similar configuration to that of the Blip-base
[Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import BlipVisionConfig, BlipVisionModel

>>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
>>> configuration = BlipVisionConfig()

>>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
>>> model = BlipVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

image_size (`int`, *optional*, defaults to 384) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 16) : The size (resolution) of each patch.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-5) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 1e-10) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

## BlipProcessor[[transformers.BlipProcessor]]

#### transformers.BlipProcessor[[transformers.BlipProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/processing_blip.py#L42)

Constructs a BLIP processor which wraps a BERT tokenizer and BLIP image processor into a single processor.

[BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) and [BertTokenizerFast](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer). See the
docstring of `__call__()` and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor (`BlipImageProcessor`) : An instance of [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.

tokenizer (`BertTokenizerFast`) : An instance of ['BertTokenizerFast`]. The tokenizer is a required input.

## BlipImageProcessor[[transformers.BlipImageProcessor]]

#### transformers.BlipImageProcessor[[transformers.BlipImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/image_processing_blip.py#L46)

Constructs a BLIP image processor.

preprocesstransformers.BlipImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/image_processing_blip.py#L159[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "do_convert_rgb", "val": ": typing.Optional[bool] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Controls the size of the image after `resize`. The shortest edge of the image is resized to
  `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
  is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
  edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
- **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image values between [0 - 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean to normalize the image by if `do_normalize` is set to `True`.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
- **do_convert_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) --
  Whether to convert the image to RGB.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - Unset: Use the channel dimension format of the input image.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Preprocess an image or batch of images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in the `preprocess` method.

size (`dict`, *optional*, defaults to `{"height" : 384, "width": 384}`): Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.

resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be overridden by the `resample` parameter in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.

do_convert_rgb (`bool`, *optional*, defaults to `True`) : Whether to convert the image to RGB.

## BlipImageProcessorFast[[transformers.BlipImageProcessorFast]]

#### transformers.BlipImageProcessorFast[[transformers.BlipImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/image_processing_blip_fast.py#L23)

Constructs a fast Blip image processor.

preprocesstransformers.BlipImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L839[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "*args", "val": ""}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.processing_utils.ImagesKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_convert_rgb** (`bool`, *optional*) --
  Whether to convert the image to RGB.
- **do_resize** (`bool`, *optional*) --
  Whether to resize the image.
- **size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Describes the maximum input dimensions to the model.
- **crop_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Size of the output image after applying `center_crop`.
- **resample** (`Annotated[Union[PILImageResampling, int, NoneType], None]`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*) --
  Whether to rescale the image.
- **rescale_factor** (`float`, *optional*) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*) --
  Whether to normalize the image.
- **image_mean** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
- **image_std** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
- **do_pad** (`bool`, *optional*) --
  Whether to pad the image. Padding is done either to the largest size in the batch
  or to a fixed square size per image. The exact padding strategy depends on the model.
- **pad_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch. Applied only when `do_pad=True.`
- **do_center_crop** (`bool`, *optional*) --
  Whether to center crop the image.
- **data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
- **input_data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
- **device** (`Annotated[Union[str, torch.device, NoneType], None]`) --
  The device to process the images on. If unset, the device is inferred from the input images.
- **return_tensors** (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) --
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
- **disable_grouping** (`bool`, *optional*) --
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
- **image_seq_length** (`int`, *optional*) --
  The number of image tokens to be used for each image in the input.
  Added for backward compatibility but this should be set as a processor attribute in future models.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

do_convert_rgb (`bool`, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *optional*) : Whether to resize the image.

size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Describes the maximum input dimensions to the model.

crop_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[PILImageResampling, int, NoneType], None]`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *optional*) : Whether to center crop the image.

data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`) : The device to process the images on. If unset, the device is inferred from the input images.

return_tensors (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) : Returns stacked tensors if set to `pt, otherwise returns a list of tensors.

disable_grouping (`bool`, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## BlipModel[[transformers.BlipModel]]

`BlipModel` is going to be deprecated in future versions, please use `BlipForConditionalGeneration`, `BlipForImageTextRetrieval` or `BlipForQuestionAnswering` depending on your usecase.

#### transformers.BlipModel[[transformers.BlipModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L525)

This model is going to be deprecated in future versions. Please use `BlipForConditionalGeneration`, `BlipForQuestionAnswering` or `BlipForImageTextRetrieval` depending on your usecase.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.BlipModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L692[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **return_loss** (`bool`, *optional*) --
  Whether or not to return the contrastive loss.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0`transformers.models.blip.modeling_blip.BlipOutput` or `tuple(torch.FloatTensor)`A `transformers.models.blip.modeling_blip.BlipOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [BlipTextModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [BlipTextModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel).
The [BlipModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipModel

>>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

**Parameters:**

config ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.blip.modeling_blip.BlipOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.blip.modeling_blip.BlipOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [BlipTextModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [BlipTextModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel).
#### get_text_features[[transformers.BlipModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L570)

Examples:

```python
>>> from transformers import AutoProcessor, BlipModel

>>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [BlipTextModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipTextModel).
#### get_image_features[[transformers.BlipModel.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L604)

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipModel

>>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

**Returns:**

`image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The image embeddings obtained by
applying the projection layer to the pooled output of [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel).

## BlipTextModel[[transformers.BlipTextModel]]

#### transformers.BlipTextModel[[transformers.BlipTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip_text.py#L509)

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in [Attention is
all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an
`encoder_hidden_states` is then expected as an input to the forward pass.

forwardtransformers.BlipTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip_text.py#L596[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "is_decoder", "val": ": typing.Optional[bool] = False"}, {"name": "cache_position", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]

encoder_hidden_states  (`torch.FloatTensor`, *optional*):
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.
encoder_attention_mask (`torch.FloatTensor`, *optional*):
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
- 1 for tokens that are **not masked**,
- 0 for tokens that are **masked**.
past_key_values (`Cache`, *optional*):
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
`decoder_input_ids` of shape `(batch_size, sequence_length)`.
use_cache (`bool`, *optional*):
If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
`past_key_values`).

## BlipTextLMHeadModel[[transformers.BlipTextLMHeadModel]]

#### transformers.BlipTextLMHeadModel[[transformers.BlipTextLMHeadModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip_text.py#L730)

forwardtransformers.BlipTextLMHeadModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip_text.py#L756[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "return_logits", "val": ": typing.Optional[bool] = False"}, {"name": "is_decoder", "val": ": typing.Optional[bool] = True"}, {"name": "reduction", "val": ": typing.Optional[str] = 'mean'"}, {"name": "cache_position", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ""}]

encoder_hidden_states (`torch.FloatTensor`, *optional*): Sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is
configured as a decoder.
encoder_attention_mask (`torch.FloatTensor`, *optional*):
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
- 1 for tokens that are **not masked**,
- 0 for tokens that are **masked**.
labels (`torch.LongTensor`, *optional*):
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
`[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
past_key_values (`Cache`, *optional*):
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
`decoder_input_ids` of shape `(batch_size, sequence_length)`.
use_cache (`bool`, *optional*):
If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
`past_key_values`).

## BlipVisionModel[[transformers.BlipVisionModel]]

#### transformers.BlipVisionModel[[transformers.BlipVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L467)

forwardtransformers.BlipVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L487[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [BlipVisionModel](/docs/transformers/main/en/model_doc/blip#transformers.BlipVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## BlipForConditionalGeneration[[transformers.BlipForConditionalGeneration]]

#### transformers.BlipForConditionalGeneration[[transformers.BlipForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L781)

BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
`input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
from the text input. If no text input is provided, the decoder will start with the [BOS] token only.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.BlipForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L808[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).0`transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Language modeling loss from the text decoder.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*) -- Prediction scores of the language modeling head of the text decoder model.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*) -- The image embeddings obtained after applying the Vision Transformer model to the input image.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [BlipForConditionalGeneration](/docs/transformers/main/en/model_doc/blip#transformers.BlipForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
>>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "A picture of"

>>> inputs = processor(images=image, text=text, return_tensors="pt")

>>> outputs = model(**inputs)
```

**Parameters:**

config ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Language modeling loss from the text decoder.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*) -- Prediction scores of the language modeling head of the text decoder model.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*) -- The image embeddings obtained after applying the Vision Transformer model to the input image.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## BlipForImageTextRetrieval[[transformers.BlipForImageTextRetrieval]]

#### transformers.BlipForImageTextRetrieval[[transformers.BlipForImageTextRetrieval]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L1169)

BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
the image.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.BlipForImageTextRetrieval.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L1208[{"name": "input_ids", "val": ": LongTensor"}, {"name": "pixel_values", "val": ": FloatTensor"}, {"name": "use_itm_head", "val": ": typing.Optional[bool] = True"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **use_itm_head** (`bool`, *optional*, defaults to `True`) --
  Whether or not to use the image-text matching head.
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0`transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss from the text decoder.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) -- The image embeddings obtained by applying the projection layer to the pooler_output.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [BlipForImageTextRetrieval](/docs/transformers/main/en/model_doc/blip#transformers.BlipForImageTextRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipForImageTextRetrieval

>>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "an image of a cat"

>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> outputs = model(**inputs)
```

**Parameters:**

config ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss from the text decoder.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) -- The image embeddings obtained by applying the projection layer to the pooler_output.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## BlipForQuestionAnswering[[transformers.BlipForQuestionAnswering]]

#### transformers.BlipForQuestionAnswering[[transformers.BlipForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L950)

BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
with the encoding of the image, and the text decoder will output the answer to the question.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.BlipForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/modeling_blip.py#L978[{"name": "input_ids", "val": ": LongTensor"}, {"name": "pixel_values", "val": ": FloatTensor"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([BlipProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0`transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss from the text decoder.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) -- The image embeddings obtained by applying the projection layer to the pooler_output.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [BlipForQuestionAnswering](/docs/transformers/main/en/model_doc/blip#transformers.BlipForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, BlipForQuestionAnswering

>>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # training
>>> text = "How many cats are in the picture?"
>>> label = "2"
>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> labels = processor(text=label, return_tensors="pt").input_ids

>>> inputs["labels"] = labels
>>> outputs = model(**inputs)
>>> loss = outputs.loss
>>> loss.backward()

>>> # inference
>>> text = "How many cats are in the picture?"
>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(processor.decode(outputs[0], skip_special_tokens=True))
2
```

**Parameters:**

config ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.blip.modeling_blip.BlipTextVisionModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BlipConfig](/docs/transformers/main/en/model_doc/blip#transformers.BlipConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss from the text decoder.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) -- The image embeddings obtained by applying the projection layer to the pooler_output.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
