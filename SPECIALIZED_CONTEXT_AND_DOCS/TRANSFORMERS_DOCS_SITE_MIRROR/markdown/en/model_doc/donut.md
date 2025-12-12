# Donut

[Donut (Document Understanding Transformer)](https://huggingface.co/papers/2111.15664) is a visual document understanding model that doesn't require an Optical Character Recognition (OCR) engine. Unlike traditional approaches that extract text using OCR before processing, Donut employs an end-to-end Transformer-based architecture to directly analyze document images. This eliminates OCR-related inefficiencies making it more accurate and adaptable to diverse languages and formats.

Donut features vision encoder ([Swin](./swin)) and a text decoder ([BART](./bart)). Swin converts document images into embeddings and BART processes them into meaningful text sequences.

You can find all the original Donut checkpoints under the [Naver Clova Information Extraction](https://huggingface.co/naver-clova-ix) organization.

> [!TIP]
> Click on the Donut models in the right sidebar for more examples of how to apply Donut to different language and vision tasks.

The examples below demonstrate how to perform document understanding tasks using Donut with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) and [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel)

```py
# pip install datasets
import torch
from transformers import pipeline
from PIL import Image

pipeline = pipeline(
    task="document-question-answering",
    model="naver-clova-ix/donut-base-finetuned-docvqa",
    device=0,
    dtype=torch.float16
)
dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]

pipeline(image=image, question="What time is the coffee break?")
```

```py
# pip install datasets
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForImageTextToText.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
question = "What time is the coffee break?"
task_prompt = f"{question}"
inputs = processor(image, task_prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs.input_ids,
    pixel_values=inputs.pixel_values,
    max_length=512
)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install datasets torchao
import torch
from datasets import load_dataset
from transformers import TorchAoConfig, AutoProcessor, AutoModelForImageTextToText

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForImageTextToText.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", quantization_config=quantization_config)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
question = "What time is the coffee break?"
task_prompt = f"{question}"
inputs = processor(image, task_prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs.input_ids,
    pixel_values=inputs.pixel_values,
    max_length=512
)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Notes

- Use Donut for document image classification as shown below.

    ```py
    >>> import re
    >>> from transformers import DonutProcessor, VisionEncoderDecoderModel
    >>> from accelerate import Accelerator
    >>> from datasets import load_dataset
    >>> import torch

    >>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    >>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

    >>> device = Accelerator().device
    >>> model.to(device)  # doctest: +IGNORE_RESULT

    >>> # load document image
    >>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    >>> image = dataset[1]["image"]

    >>> # prepare decoder inputs
    >>> task_prompt = ""
    >>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    >>> pixel_values = processor(image, return_tensors="pt").pixel_values

    >>> outputs = model.generate(
    ...     pixel_values.to(device),
    ...     decoder_input_ids=decoder_input_ids.to(device),
    ...     max_length=model.decoder.config.max_position_embeddings,
    ...     pad_token_id=processor.tokenizer.pad_token_id,
    ...     eos_token_id=processor.tokenizer.eos_token_id,
    ...     use_cache=True,
    ...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    ...     return_dict_in_generate=True,
    ... )

    >>> sequence = processor.batch_decode(outputs.sequences)[0]
    >>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    >>> sequence = re.sub(r"", "", sequence, count=1).strip()  # remove first task start token
    >>> print(processor.token2json(sequence))
    {'class': 'advertisement'}
    ```

- Use Donut for document parsing as shown below.

    ```py
    >>> import re
    >>> from accelerate import Accelerator
    >>> from datasets import load_dataset
    >>> from transformers import DonutProcessor, VisionEncoderDecoderModel
    >>> import torch

    >>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    >>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    >>> device = Accelerator().device
    >>> model.to(device)  # doctest: +IGNORE_RESULT

    >>> # load document image
    >>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    >>> image = dataset[2]["image"]

    >>> # prepare decoder inputs
    >>> task_prompt = ""
    >>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    >>> pixel_values = processor(image, return_tensors="pt").pixel_values

    >>> outputs = model.generate(
    ...     pixel_values.to(device),
    ...     decoder_input_ids=decoder_input_ids.to(device),
    ...     max_length=model.decoder.config.max_position_embeddings,
    ...     pad_token_id=processor.tokenizer.pad_token_id,
    ...     eos_token_id=processor.tokenizer.eos_token_id,
    ...     use_cache=True,
    ...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    ...     return_dict_in_generate=True,
    ... )

    >>> sequence = processor.batch_decode(outputs.sequences)[0]
    >>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    >>> sequence = re.sub(r"", "", sequence, count=1).strip()  # remove first task start token
    >>> print(processor.token2json(sequence))
    {'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': 
    {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
    ```

## DonutSwinConfig[[transformers.DonutSwinConfig]]

#### transformers.DonutSwinConfig[[transformers.DonutSwinConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/configuration_donut_swin.py#L24)

This is the configuration class to store the configuration of a [DonutSwinModel](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinModel). It is used to instantiate a
Donut model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Donut
[naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import DonutSwinConfig, DonutSwinModel

>>> # Initializing a Donut naver-clova-ix/donut-base style configuration
>>> configuration = DonutSwinConfig()

>>> # Randomly initializing a model from the naver-clova-ix/donut-base style configuration
>>> model = DonutSwinModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 4) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

embed_dim (`int`, *optional*, defaults to 96) : Dimensionality of patch embedding.

depths (`list(int)`, *optional*, defaults to `[2, 2, 6, 2]`) : Depth of each layer in the Transformer encoder.

num_heads (`list(int)`, *optional*, defaults to `[3, 6, 12, 24]`) : Number of attention heads in each layer of the Transformer encoder.

window_size (`int`, *optional*, defaults to 7) : Size of windows.

mlp_ratio (`float`, *optional*, defaults to 4.0) : Ratio of MLP hidden dimensionality to embedding dimensionality.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether or not a learnable bias should be added to the queries, keys and values.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings and encoder.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

drop_path_rate (`float`, *optional*, defaults to 0.1) : Stochastic depth rate.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

use_absolute_embeddings (`bool`, *optional*, defaults to `False`) : Whether or not to add absolute position embeddings to the patch embeddings.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

## DonutImageProcessor[[transformers.DonutImageProcessor]]

#### transformers.DonutImageProcessor[[transformers.DonutImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/image_processing_donut.py#L68)

Constructs a Donut image processor.

preprocesstransformers.DonutImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/image_processing_donut.py#L321[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_thumbnail", "val": ": typing.Optional[bool] = None"}, {"name": "do_align_long_axis", "val": ": typing.Optional[bool] = None"}, {"name": "do_pad", "val": ": typing.Optional[bool] = None"}, {"name": "random_padding", "val": ": bool = False"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": typing.Optional[transformers.image_utils.ChannelDimension] = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image after resizing. Shortest edge of the image is resized to min(size["height"],
  size["width"]) with the longest edge resized to keep the input aspect ratio.
- **resample** (`int`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
- **do_thumbnail** (`bool`, *optional*, defaults to `self.do_thumbnail`) --
  Whether to resize the image using thumbnail method.
- **do_align_long_axis** (`bool`, *optional*, defaults to `self.do_align_long_axis`) --
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
- **do_pad** (`bool`, *optional*, defaults to `self.do_pad`) --
  Whether to pad the image. If `random_padding` is set to `True`, each image is padded with a random
  amount of padding on each size, up to the largest image size in the batch. Otherwise, all images are
  padded to the largest image size in the batch.
- **random_padding** (`bool`, *optional*, defaults to `self.random_padding`) --
  Whether to use random padding when padding the image. If `True`, each image in the batch with be padded
  with a random amount of padding on each side up to the size of the largest image in the batch.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image pixel values.
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean to use for normalization.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation to use for normalization.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - Unset: defaults to the channel dimension format of the input image.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Preprocess an image or batch of images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in the `preprocess` method.

size (`dict[str, int]` *optional*, defaults to `{"shortest_edge" : 224}`): Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess` method.

resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) : Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.

do_thumbnail (`bool`, *optional*, defaults to `True`) : Whether to resize the image using thumbnail method.

do_align_long_axis (`bool`, *optional*, defaults to `False`) : Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.

do_pad (`bool`, *optional*, defaults to `True`) : Whether to pad the image. If `random_padding` is set to `True` in `preprocess`, each image is padded with a random amount of padding on each size, up to the largest image size in the batch. Otherwise, all images are padded to the largest image size in the batch.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess` method.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Image standard deviation.

## DonutImageProcessorFast[[transformers.DonutImageProcessorFast]]

#### transformers.DonutImageProcessorFast[[transformers.DonutImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/image_processing_donut_fast.py#L38)

Constructs a fast Donut image processor.

preprocesstransformers.DonutImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/image_processing_donut_fast.py#L58[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.donut.image_processing_donut.DonutImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
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
  Added for backward compatibility but this should be set as a processor attribute in future models.
- **do_thumbnail** (`bool`, *optional*, defaults to `self.do_thumbnail`) --
  Whether to resize the image using thumbnail method.
- **do_align_long_axis** (`bool`, *optional*, defaults to `self.do_align_long_axis`) --
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
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

do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`) : Whether to resize the image using thumbnail method.

do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`) : Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## DonutProcessor[[transformers.DonutProcessor]]

#### transformers.DonutProcessor[[transformers.DonutProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/processing_donut.py#L35)

Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single
processor.

[DonutProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutProcessor) offers all the functionalities of [DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor) and
[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. See the [__call__()](/docs/transformers/main/en/model_doc/donut#transformers.DonutProcessor.__call__) and
[decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

__call__transformers.DonutProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/processing_donut.py#L54[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "text", "val": ": typing.Union[str, list[str], NoneType] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.donut.processing_donut.DonutProcessorKwargs]"}]

When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
`__call__()` and returns its output. If used in the context
`~DonutProcessor.as_target_processor` this method forwards all its arguments to DonutTokenizer's
`~DonutTokenizer.__call__`. Please refer to the docstring of the above two methods for more information.

**Parameters:**

image_processor ([DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor), *optional*) : An instance of [DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor). The image processor is a required input.

tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`], *optional*) : An instance of [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. The tokenizer is a required input.
#### from_pretrained[[transformers.DonutProcessor.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1349)

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
#### save_pretrained[[transformers.DonutProcessor.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L778)

Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
can be reloaded using the [from_pretrained()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### batch_decode[[transformers.DonutProcessor.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1520)

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.
#### decode[[transformers.DonutProcessor.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1529)

This method forwards all its arguments to PreTrainedTokenizer's [decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## DonutSwinModel[[transformers.DonutSwinModel]]

#### transformers.DonutSwinModel[[transformers.DonutSwinModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/modeling_donut_swin.py#L807)

The bare Donut Swin Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.DonutSwinModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/modeling_donut_swin.py#L831[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "bool_masked_pos", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor). See [DonutImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DonutSwinConfig](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed) -- Average pooling of the last layer hidden-state.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **reshaped_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.
The [DonutSwinModel](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([DonutSwinModel](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

use_mask_token (`bool`, *optional*, defaults to `False`) : Whether to use a mask token for masked image modeling.

**Returns:**

``transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DonutSwinConfig](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed) -- Average pooling of the last layer hidden-state.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **reshaped_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.

## DonutSwinForImageClassification[[transformers.DonutSwinForImageClassification]]

#### transformers.DonutSwinForImageClassification[[transformers.DonutSwinForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/modeling_donut_swin.py#L903)

DonutSwin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

Note that it's possible to fine-tune DonutSwin on higher resolution images than the ones it has been trained on, by
setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
position embeddings to the higher resolution.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.DonutSwinForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/modeling_donut_swin.py#L918[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor). See [DonutImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DonutImageProcessor](/docs/transformers/main/en/model_doc/donut#transformers.DonutImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or `tuple(torch.FloatTensor)`A `transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DonutSwinConfig](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **reshaped_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.
The [DonutSwinForImageClassification](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, DonutSwinForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("naver-clova-ix/donut-base")
>>> model = DonutSwinForImageClassification.from_pretrained("naver-clova-ix/donut-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([DonutSwinForImageClassification](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinForImageClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DonutSwinConfig](/docs/transformers/main/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **reshaped_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.
