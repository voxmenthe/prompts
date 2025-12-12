*This model was released on {release\_date} and added to Hugging Face Transformers on 2025-08-19.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# KOSMOS-2.5

The Kosmos-2.5 model was proposed in [KOSMOS-2.5: A Multimodal Literate Model](https://huggingface.co/papers/2309.11419/) by Microsoft.

The abstract from the paper is the following:

*We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos2_5_ocr.png) ![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos2_5_md.png) Overview of tasks that KOSMOS-2.5 can handle. Taken from the [original paper](https://huggingface.co/papers/2309.11419).

The examples below demonstrates how to generate with [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), for both Markdown and OCR tasks.

AutoModel - Markdown Task

AutoModel - OCR Task


```
import re
import torch
import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration, infer_device

repo = "microsoft/kosmos-2.5"
device = "cuda:0"
dtype = torch.bfloat16
model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, device_map=device, dtype=dtype)
processor = AutoProcessor.from_pretrained(repo)

# sample image
url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<md>"
inputs = processor(text=prompt, images=image, return_tensors="pt")

height, width = inputs.pop("height"), inputs.pop("width")
raw_width, raw_height = image.size
scale_height = raw_height / height
scale_width = raw_width / width

inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])
```

## Chat version

The authors also released Kosmos-2.5 Chat, which is a chat version optimized for document understanding. You can use it like so:


```
import re
import torch
import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

repo = "microsoft/kosmos-2.5-chat"
device = "cuda:0"
dtype = torch.bfloat16

model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo,
                                                          device_map=device,
                                                          torch_dtype=dtype,
                                                          attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(repo)

# sample image
url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"

image = Image.open(requests.get(url, stream=True).raw)

question = "What is the sub total of the receipt?"
template = "<md>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
prompt = template.format(question)
inputs = processor(text=prompt, images=image, return_tensors="pt")

height, width = inputs.pop("height"), inputs.pop("width")
raw_width, raw_height = image.size
scale_height = raw_height / height
scale_width = raw_width / width

inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])
```

## Kosmos2\_5Config

### class transformers.Kosmos2\_5Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/configuration_kosmos2_5.py#L212)

( text\_config = None vision\_config = None latent\_query\_num = 2048 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize `Kosmos2_5TextConfig`.
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize `Kosmos2_5VisionConfig`.
* **latent\_query\_num** (`int`, *optional*, defaults to 2048) —
  The number of latent query tokens that represent the image features used in the text decoder component.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

This is the configuration class to store the configuration of a [Kosmos2\_5Model](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Model). It is used to instantiate a
KOSMOS-2.5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the KOSMOS-2.5
[microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

## Kosmos2\_5ImageProcessor

### class transformers.Kosmos2\_5ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/image_processing_kosmos2_5.py#L76)

( do\_convert\_rgb: bool = True do\_normalize: bool = True patch\_size: typing.Optional[dict[str, int]] = None max\_patches: int = 4096 \*\*kwargs  )

Parameters

* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. According to Kosmos2\_5 paper and code, the image is normalized with its own mean and standard
  deviation.
* **patch\_size** (`Dict[str, int]`, *optional*, defaults to `{"height" -- 16, "width": 16}`):
  The patch size to use for the image. According to Kosmos2\_5 paper and code, the patch size is 16x16.
* **max\_patches** (`int`, *optional*, defaults to 4096) —
  The maximum number of patches to extract from the image as per the
  [KOSMOS 2.5 paper](https://huggingface.co/papers/2309.11419).

Constructs a Kosmos2\_5 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/image_processing_kosmos2_5.py#L241)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_convert\_rgb: typing.Optional[bool] = None do\_normalize: typing.Optional[bool] = None max\_patches: typing.Optional[int] = None patch\_size: typing.Optional[dict[str, int]] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **max\_patches** (`int`, *optional*, defaults to `self.max_patches`) —
  Maximum number of patches to extract.
* **patch\_size** (`dict`, *optional*, defaults to `self.patch_size`) —
  Dictionary containing the patch height and width.
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

Preprocess an image or batch of images. The processor first computes the maximum possible number of
aspect-ratio preserving patches of size `patch_size` that can be extracted from the image. It then pads the
image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the
images are standardized following the tensorflow implementation of `per_image_standardization`
(<https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization>).

## Kosmos2\_5ImageProcessorFast

### class transformers.Kosmos2\_5ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/image_processing_kosmos2_5_fast.py#L76)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.kosmos2\_5.image\_processing\_kosmos2\_5\_fast.Kosmos2\_5FastImageProcessorKwargs]  )

Constructs a fast Kosmos2 5 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/image_processing_kosmos2_5_fast.py#L89)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.kosmos2\_5.image\_processing\_kosmos2\_5\_fast.Kosmos2\_5FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **patch\_size** (`Dict[str, int]`, *optional*, defaults to `{"height" -- 16, "width": 16}`):
  The patch size to use for the image. According to Kosmos2\_5 paper and code, the patch size is 16x16.
* **max\_patches** (`int`, *optional*, defaults to 4096) —
  The maximum number of patches to extract from the image as per the
  [KOSMOS 2.5 paper](https://huggingface.co/papers/2309.11419).

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Kosmos2\_5Processor

### class transformers.Kosmos2\_5Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/processing_kosmos2_5.py#L55)

( image\_processor tokenizer  )

Parameters

* **image\_processor** (`Kosmos2_5ImageProcessor`) —
  An instance of [Kosmos2\_5ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ImageProcessor). The image processor is a required input.
* **tokenizer** (Union[`T5TokenizerFast`, `T5Tokenizer`]) —
  An instance of [‘T5TokenizerFast`] or [‘T5Tokenizer`]. The tokenizer is a required input.

Constructs a Kosmos2\_5 processor which wraps a PreTrainedTokenizerFast and Kosmos2\_5 image processor into a single
processor.

[Kosmos2\_5Processor](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Processor) offers all the functionalities of [Kosmos2\_5ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ImageProcessor) and [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast). See
the docstring of `__call__()` and [decode()](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Processor.decode) for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/processing_kosmos2_5.py#L143)

( \*args \*\*kwargs  )

This method forwards all its arguments to Kosmos2\_5TokenizerFast’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode).
Please refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/processing_kosmos2_5.py#L150)

( \*args \*\*kwargs  )

This method forwards all its arguments to Kosmos2\_5TokenizerFast’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please
refer to the docstring of this method for more information.

## Kosmos2\_5Model

### class transformers.Kosmos2\_5Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/modeling_kosmos2_5.py#L1390)

( config: Kosmos2\_5Config  )

Parameters

* **config** ([Kosmos2\_5Config](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Config)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

KOSMOS-2.5 Model for generating text and image features. The model consists of a vision encoder and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/modeling_kosmos2_5.py#L1409)

( input\_ids: typing.Optional[torch.Tensor] = None flattened\_patches: typing.Optional[torch.Tensor] = None width: typing.Optional[torch.Tensor] = None height: typing.Optional[torch.Tensor] = None image\_embeds\_position\_mask: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None image\_embeds: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.kosmos2_5.modeling_kosmos2_5.Kosmos2_5ModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **flattened\_patches** (`torch.FloatTensor` of shape `(batch_size, max_patches, 2 + patch_height * patch_width * image_channels)`) —
  Flattened patches of the images. `flattened_patches` can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See
  [Kosmos2\_5ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **width** (`torch.FloatTensor` of shape `(batch_size,)`) —
  The original width (before resizing) of each image in the batch. This can be obtained using
  [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [Kosmos2\_5ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **height** (`torch.FloatTensor` of shape `(batch_size,)`) —
  The original height (before resizing) of each image in the batch. This can be obtained using
  [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [Kosmos2\_5ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **image\_embeds\_position\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the location in a sequence to insert the image features . Mask values selected in `[0, 1]`:
  + 1 for places where to put the image features,
  + 0 for places that are not for image features (i.e. for text tokens).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **image\_embeds** — (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
  Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.kosmos2_5.modeling_kosmos2_5.Kosmos2_5ModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.kosmos2_5.modeling_kosmos2_5.Kosmos2_5ModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`<class 'transformers.models.kosmos2_5.configuration_kosmos2_5.Kosmos2_5Config'>`) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **width** (`torch.FloatTensor` of shape `(batch_size,)`) — The original width (before resizing) of each image in the batch.
* **height** (`torch.FloatTensor` of shape `(batch_size,)`) — The original height (before resizing) of each image in the batch.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*) — Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **projection\_attentions** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
  the weighted average in the self-attention heads.
* **vision\_model\_output(`BaseModelOutputWithPooling`,** *optional*) — The output of the `Kosmos2VisionModel`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

The [Kosmos2\_5Model](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Kosmos2_5Model

>>> model = Kosmos2_5Model.from_pretrained("microsoft/kosmos2.5")
>>> processor = AutoProcessor.from_pretrained("microsoft/kosmos2.5")

>>> url = "https://huggingface.co/microsoft/kosmos2.5/resolve/main/snowman.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> text = (
...     "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863>"
...     "</object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911>"
...     "</object>"
... )

>>> inputs = processor(text=text, images=image, return_tensors="pt", add_eos_token=True)

>>> last_hidden_state = model(
...     pixel_values=inputs["pixel_values"],
...     input_ids=inputs["input_ids"],
...     attention_mask=inputs["attention_mask"],
...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
... ).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 91, 2048]
```

## Kosmos2\_5ForConditionalGeneration

### class transformers.Kosmos2\_5ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/modeling_kosmos2_5.py#L1669)

( config: Kosmos2\_5Config  )

Parameters

* **config** ([Kosmos2\_5Config](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Config)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

KOSMOS-2.5 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2_5/modeling_kosmos2_5.py#L1693)

( input\_ids: typing.Optional[torch.Tensor] = None flattened\_patches: typing.Optional[torch.Tensor] = None width: typing.Optional[torch.Tensor] = None height: typing.Optional[torch.Tensor] = None image\_embeds\_position\_mask: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None image\_embeds: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.kosmos2_5.modeling_kosmos2_5.Kosmos2_5ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **flattened\_patches** (`torch.FloatTensor` of shape `(batch_size, max_patches, 2 + patch_height * patch_width * image_channels)`) —
  Flattened patches of the images. `flattened_patches` can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See
  [Kosmos2\_5ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **width** (`torch.FloatTensor` of shape `(batch_size,)`) —
  The original width (before resizing) of each image in the batch. This can be obtained using
  [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [Kosmos2\_5ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **height** (`torch.FloatTensor` of shape `(batch_size,)`) —
  The original height (before resizing) of each image in the batch. This can be obtained using
  [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [Kosmos2\_5ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **image\_embeds\_position\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the location in a sequence to insert the image features . Mask values selected in `[0, 1]`:
  + 1 for places where to put the image features,
  + 0 for places that are not for image features (i.e. for text tokens).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **image\_embeds** — (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*):
  Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

Returns

`transformers.models.kosmos2_5.modeling_kosmos2_5.Kosmos2_5ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.kosmos2_5.modeling_kosmos2_5.Kosmos2_5ForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`<class 'transformers.models.kosmos2_5.configuration_kosmos2_5.Kosmos2_5Config'>`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **width** (`torch.FloatTensor` of shape `(batch_size,)`) — The original width (before resizing) of each image in the batch.
* **height** (`torch.FloatTensor` of shape `(batch_size,)`) — The original height (before resizing) of each image in the batch.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*) — Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **projection\_attentions** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
  the weighted average in the self-attention heads.
* **vision\_model\_output(`BaseModelOutputWithPooling`,** *optional*) — The output of the `Kosmos2VisionModel`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

The [Kosmos2\_5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> import torch
>>> from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

>>> repo = "microsoft/kosmos-2.5"
>>> device = "cuda:0"
>>> dtype = torch.bfloat16 # torch.float16
>>> model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, device_map=device, dtype=dtype)
>>> processor = AutoProcessor.from_pretrained(repo)

>>> url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"

>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "<ocr>" # <md>

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")
>>> height, width = inputs.pop("height"), inputs.pop("width")
>>> inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
>>> inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)

>>> generated_ids = model.generate(**inputs,max_new_tokens=1024)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> generated_text
'<ocr><bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_612></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_650></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_644></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_687></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/kosmos2_5.md)
