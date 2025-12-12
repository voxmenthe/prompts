*This model was released on 2021-11-30 and added to Hugging Face Transformers on 2022-08-12.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Donut

[Donut (Document Understanding Transformer)](https://huggingface.co/papers/2111.15664) is a visual document understanding model that doesn’t require an Optical Character Recognition (OCR) engine. Unlike traditional approaches that extract text using OCR before processing, Donut employs an end-to-end Transformer-based architecture to directly analyze document images. This eliminates OCR-related inefficiencies making it more accurate and adaptable to diverse languages and formats.

Donut features vision encoder ([Swin](./swin)) and a text decoder ([BART](./bart)). Swin converts document images into embeddings and BART processes them into meaningful text sequences.

You can find all the original Donut checkpoints under the [Naver Clova Information Extraction](https://huggingface.co/naver-clova-ix) organization.

Click on the Donut models in the right sidebar for more examples of how to apply Donut to different language and vision tasks.

The examples below demonstrate how to perform document understanding tasks using Donut with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) and [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel)

Pipeline

AutoModel


```
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

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
# pip install datasets torchao
import torch
from datasets import load_dataset
from transformers import TorchAoConfig, AutoProcessor, AutoModelForVision2Seq

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", quantization_config=quantization_config)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
question = "What time is the coffee break?"
task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
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

* Use Donut for document image classification as shown below.


  ```
  >>> import re
  >>> from transformers import DonutProcessor, VisionEncoderDecoderModel, infer_device
  >>> from datasets import load_dataset
  >>> import torch

  >>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
  >>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

  >>> device = infer_device()
  >>> model.to(device)  # doctest: +IGNORE_RESULT

  >>> # load document image
  >>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
  >>> image = dataset[1]["image"]

  >>> # prepare decoder inputs
  >>> task_prompt = "<s_rvlcdip>"
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
  >>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
  >>> print(processor.token2json(sequence))
  {'class': 'advertisement'}
  ```
* Use Donut for document parsing as shown below.


  ```
  >>> import re
  >>> from transformers import DonutProcessor, VisionEncoderDecoderModel, infer_device
  >>> from datasets import load_dataset
  >>> import torch

  >>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
  >>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

  >>> device = infer_device()
  >>> model.to(device)  # doctest: +IGNORE_RESULT

  >>> # load document image
  >>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
  >>> image = dataset[2]["image"]

  >>> # prepare decoder inputs
  >>> task_prompt = "<s_cord-v2>"
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
  >>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
  >>> print(processor.token2json(sequence))
  {'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': 
  {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
  ```

## DonutSwinConfig

### class transformers.DonutSwinConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/configuration_donut_swin.py#L24)

( image\_size = 224 patch\_size = 4 num\_channels = 3 embed\_dim = 96 depths = [2, 2, 6, 2] num\_heads = [3, 6, 12, 24] window\_size = 7 mlp\_ratio = 4.0 qkv\_bias = True hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 drop\_path\_rate = 0.1 hidden\_act = 'gelu' use\_absolute\_embeddings = False initializer\_range = 0.02 layer\_norm\_eps = 1e-05 \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 4) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **embed\_dim** (`int`, *optional*, defaults to 96) —
  Dimensionality of patch embedding.
* **depths** (`list(int)`, *optional*, defaults to `[2, 2, 6, 2]`) —
  Depth of each layer in the Transformer encoder.
* **num\_heads** (`list(int)`, *optional*, defaults to `[3, 6, 12, 24]`) —
  Number of attention heads in each layer of the Transformer encoder.
* **window\_size** (`int`, *optional*, defaults to 7) —
  Size of windows.
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
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

This is the configuration class to store the configuration of a [DonutSwinModel](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinModel). It is used to instantiate a
Donut model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Donut
[naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DonutSwinConfig, DonutSwinModel

>>> # Initializing a Donut naver-clova-ix/donut-base style configuration
>>> configuration = DonutSwinConfig()

>>> # Randomly initializing a model from the naver-clova-ix/donut-base style configuration
>>> model = DonutSwinModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DonutImageProcessor

### class transformers.DonutImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/image_processing_donut.py#L55)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_thumbnail: bool = True do\_align\_long\_axis: bool = False do\_pad: bool = True do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by
  `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 224}`):
  Size of the image after resizing. The shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
  method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_thumbnail** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image using thumbnail method.
* **do\_align\_long\_axis** (`bool`, *optional*, defaults to `False`) —
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image. If `random_padding` is set to `True` in `preprocess`, each image is padded with a
  random amount of padding on each size, up to the largest image size in the batch. Otherwise, all images are
  padded to the largest image size in the batch.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Image standard deviation.

Constructs a Donut image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/image_processing_donut.py#L311)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_thumbnail: typing.Optional[bool] = None do\_align\_long\_axis: typing.Optional[bool] = None do\_pad: typing.Optional[bool] = None random\_padding: bool = False do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. Shortest edge of the image is resized to min(size[“height”],
  size[“width”]) with the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_thumbnail** (`bool`, *optional*, defaults to `self.do_thumbnail`) —
  Whether to resize the image using thumbnail method.
* **do\_align\_long\_axis** (`bool`, *optional*, defaults to `self.do_align_long_axis`) —
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the image. If `random_padding` is set to `True`, each image is padded with a random
  amount of padding on each size, up to the largest image size in the batch. Otherwise, all images are
  padded to the largest image size in the batch.
* **random\_padding** (`bool`, *optional*, defaults to `self.random_padding`) —
  Whether to use random padding when padding the image. If `True`, each image in the batch with be padded
  with a random amount of padding on each side up to the size of the largest image in the batch.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image pixel values.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization.
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
  + Unset: defaults to the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## DonutImageProcessorFast

### class transformers.DonutImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/image_processing_donut_fast.py#L64)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.donut.image\_processing\_donut\_fast.DonutFastImageProcessorKwargs]  )

Constructs a fast Donut image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/image_processing_donut_fast.py#L84)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.donut.image\_processing\_donut\_fast.DonutFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **do\_thumbnail** (`bool`, *optional*, defaults to `self.do_thumbnail`) —
  Whether to resize the image using thumbnail method.
* **do\_align\_long\_axis** (`bool`, *optional*, defaults to `self.do_align_long_axis`) —
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the image. If `random_padding` is set to `True`, each image is padded with a random
  amount of padding on each size, up to the largest image size in the batch. Otherwise, all images are
  padded to the largest image size in the batch.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## DonutFeatureExtractor

### class transformers.DonutFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/feature_extraction_donut.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## DonutProcessor

### class transformers.DonutProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/processing_donut.py#L37)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** ([DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor), *optional*) —
  An instance of [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor). The image processor is a required input.
* **tokenizer** ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`], *optional*) —
  An instance of [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. The tokenizer is a required input.

Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single
processor.

[DonutProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutProcessor) offers all the functionalities of [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor) and
[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. See the [**call**()](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutProcessor.__call__) and
[decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/processing_donut.py#L77)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None text: typing.Union[str, list[str], NoneType] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.donut.processing\_donut.DonutProcessorKwargs]  )

When used in normal mode, this method forwards all its arguments to AutoImageProcessor’s
`__call__()` and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to DonutTokenizer’s
`~DonutTokenizer.__call__`. Please refer to the docstring of the above two methods for more information.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1272)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* \***\*kwargs** —
  Additional keyword arguments passed along to both
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and
  `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L653)

( save\_directory push\_to\_hub: bool = False legacy\_serialization: bool = True \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
  be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **legacy\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to save processor attributes in separate config files (legacy) or in processor’s config
  file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
  Set to `legacy_serialization=True` until then.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it
can be reloaded using the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## DonutSwinModel

### class transformers.DonutSwinModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/modeling_donut_swin.py#L856)

( config add\_pooling\_layer = True use\_mask\_token = False  )

Parameters

* **config** ([DonutSwinModel](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer
* **use\_mask\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to use a mask token for masked image modeling.

The bare Donut Swin Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/modeling_donut_swin.py#L888)

( pixel\_values: typing.Optional[torch.FloatTensor] = None bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor). See [DonutImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor) for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
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
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DonutSwinConfig](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed) — Average pooling of the last layer hidden-state.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **reshaped\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.

The [DonutSwinModel](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## DonutSwinForImageClassification

### class transformers.DonutSwinForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/modeling_donut_swin.py#L968)

( config  )

Parameters

* **config** ([DonutSwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DonutSwin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

Note that it’s possible to fine-tune DonutSwin on higher resolution images than the ones it has been trained on, by
setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
position embeddings to the higher resolution.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/donut/modeling_donut_swin.py#L983)

( pixel\_values: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor). See [DonutImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.donut.modeling_donut_swin.DonutSwinImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DonutSwinConfig](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **reshaped\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.

The [DonutSwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/donut.md)
