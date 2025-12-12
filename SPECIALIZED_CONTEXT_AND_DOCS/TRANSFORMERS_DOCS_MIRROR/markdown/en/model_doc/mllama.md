*This model was released on 2024-09-25 and added to Hugging Face Transformers on 2024-09-25.*

# Mllama

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The [Llama 3.2-Vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image.

**Model Architecture:** Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. To support image recognition tasks, the Llama 3.2-Vision model uses a separately trained vision adapter that integrates with the pre-trained Llama 3.1 language model. The adapter consists of a series of cross-attention layers that feed image encoder representations into the core LLM.

## Usage Tips

* For image+text and text inputs use `MllamaForConditionalGeneration`.
* For text-only inputs use `MllamaForCausalLM` for generation to avoid loading vision tower.
* Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images across samples and to a maximum number of tiles within each image.
* The text passed to the processor should have the `"<|image|>"` tokens where the images should be inserted.
* The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as text to the processor. If you’re using `transformers>=4.49.0`, you can also get a vectorized output from `apply_chat_template`. See the **Usage Examples** below for more details on how to use it.

Mllama has an extra token used as a placeholder for image positions in the text. It means that input ids and an input embedding layer will have an extra token. But since the weights for input and output embeddings are not tied, the `lm_head` layer has one less token and will fail if you want to calculate loss on image tokens or apply some logit processors. In case you are training, make sure to mask out special `"<|image|>"` tokens in the `labels` as the model should not be trained on predicting them.

Otherwise if you see CUDA-side index errors when generating, use the below code to expand the `lm_head` by one more token.


```
old_embeddings = model.get_output_embeddings()

num_tokens = model.vocab_size + 1
resized_embeddings = model._get_resized_lm_head(old_embeddings, new_num_tokens=num_tokens, mean_resizing=True)
resized_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
model.set_output_embeddings(resized_embeddings)
```

## Usage Example

#### Instruct model


```
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    [
        {
            "role": "user", 
            "content": [
                {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                {"type": "text", "text": "What does the image show?"}
            ]
        }
    ],
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=25)
print(processor.decode(output[0]))
```

#### Base model


```
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "<|image|>If I had to write a haiku for this one"
url = "https://llava-vl.github.io/static/images/view.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=25)
print(processor.decode(output[0], skip_special_tokens=True))
```

## MllamaConfig

### class transformers.MllamaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/configuration_mllama.py#L299)

( vision\_config = None text\_config = None image\_token\_index = 128256 \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaTextConfig`) —
  The config object or dictionary of the text backbone.
* **image\_token\_index** (`int`, *optional*, defaults to 128256) —
  The image token index to encode the image prompt.

This is the configuration class to store the configuration of a [MllamaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration). It is used to instantiate an
Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Mllama-9B.

e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MllamaForConditionalGeneration, MllamaConfig, MllamaVisionConfig, MllamaTextConfig

>>> # Initializing a CLIP-vision config
>>> vision_config = MllamaVisionConfig()

>>> # Initializing a Llama config
>>> text_config = MllamaTextConfig()

>>> # Initializing a mllama-11b style configuration
>>> configuration = MllamaConfig(vision_config, text_config)

>>> # Initializing a model from the mllama-11b style configuration
>>> model = MllamaForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MllamaProcessor

### class transformers.MllamaProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/processing_mllama.py#L175)

( image\_processor tokenizer chat\_template = None  )

Parameters

* **image\_processor** ([MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor)) —
  The image processor is a required input.
* **tokenizer** ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`]) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a Mllama processor which wraps [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor) and
`PretrainedTokenizerFast` into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more
information.
The preferred way of passing kwargs is as a dictionary per modality, see usage example below.


```
from transformers import MllamaProcessor
from PIL import Image

processor = MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

processor(
    images=your_pil_image,
    text=["<|image|>If I had to write a haiku for this one"],
    images_kwargs = {"size": {"height": 448, "width": 448}},
    text_kwargs = {"padding": "right"},
    common_kwargs = {"return_tensors": "pt"},
)
```

#### post\_process\_image\_text\_to\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/processing_mllama.py#L344)

( generated\_outputs skip\_special\_tokens = True clean\_up\_tokenization\_spaces = False \*\*kwargs  ) → `list[str]`

Parameters

* **generated\_outputs** (`torch.Tensor` or `np.ndarray`) —
  The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
  or `(sequence_length,)`.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to remove special tokens in the output. Argument passed to the tokenizer’s `batch_decode` method.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `False`) —
  Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer’s `batch_decode` method.
* \***\*kwargs** —
  Additional arguments to be passed to the tokenizer’s `batch_decode method`.

Returns

`list[str]`

The decoded text.

Post-process the output of the model to decode the text.

## MllamaImageProcessor

### class transformers.MllamaImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L536)

( do\_convert\_rgb: bool = True do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: bool = True max\_image\_tiles: int = 4 \*\*kwargs  )

Parameters

* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
  Only has an effect if the input image is in the PIL format.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image tile. Should be a dictionary containing ‘height’ and ‘width’ keys, both with integer values.
  The height and width values should be equal.
* **resample** (`int`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to 0.0) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether or not to pad the images to the largest height and width in the batch.
* **max\_image\_tiles** (`int`, *optional*, defaults to 4) —
  The maximum number of tiles to split the image into.

Constructs a Mllama image processor.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L789)

( image: ndarray size: dict aspect\_ratio: tuple data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Size of the output image.
* **aspect\_ratio** (`tuple[int, int]`) —
  The aspect ratio of the image.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the image. If not provided, it will be the same as the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.

Returns

`np.ndarray`

The padded image.

Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
(1, 2), the image will be padded to 224x448.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L601)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_convert\_rgb: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: typing.Optional[PIL.Image.Resampling] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None max\_image\_tiles: typing.Optional[int] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None  ) → `BatchFeature` of the following structure

Parameters

* **images** (`ImageInput`) —
  A list of images to preprocess.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image tile. Should be a dictionary containing ‘height’ and ‘width’ keys, both with integer values.
  The height and width values should be equal.
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
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether or not to pad the images to the largest height and width in the batch.
* **max\_image\_tiles** (`int`, *optional*, defaults to `self.max_image_tiles`) —
  The maximum number of tiles to split the image into.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.

Returns

`BatchFeature` of the following structure

* **pixel\_values** (`TensorType`): The preprocessed pixel values.
* **aspect\_ratio\_ids** (`TensorType`): The aspect ratio ids of the images.
* **num\_tiles** (`list[list[int]]`): The number of tiles for each image in the batch.

Preprocess a batch of images.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L836)

( image: ndarray size: dict max\_image\_tiles: int resample: Resampling = <Resampling.BILINEAR: 2> data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  ) → `Union[np.ndarray, tuple[int, int]]`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Size of the output image.
* **max\_image\_tiles** (`int`) —
  The maximum number of tiles to split the image into.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  Resampling filter to use when resizing the image.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the image. If not provided, it will be the same as the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.

Returns

`Union[np.ndarray, tuple[int, int]]`

The resized image and a tuple containing the number of tiles
along the height and width dimensions.

Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
The optimal canvas size is calculated based on the maximum number of tiles and the tile size.

The function first determines the best tile arrangement for the image, then resizes the image
to fit within this canvas. The resized image and the number of tiles along the height and width
dimensions are returned.

## MllamaForConditionalGeneration

### class transformers.MllamaForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1541)

( config: MllamaConfig  )

Parameters

* **config** ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mllama model which consists of a vision encoder and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1577)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None aspect\_ratio\_mask: typing.Optional[torch.Tensor] = None aspect\_ratio\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None cross\_attention\_mask: typing.Optional[torch.Tensor] = None cross\_attention\_states: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor). See [MllamaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MllamaProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor) uses
  [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor) for processing images).
* **aspect\_ratio\_mask** (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*) —
  Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:
  + 1 for tiles that are **not masked**,
  + 0 for tiles that are **masked**.
* **aspect\_ratio\_ids** (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*) —
  Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
  These ids correspond to indices in the model’s list of supported aspect ratios, offset by 1.

  For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:

  + An image with aspect ratio [1, 1] would have ID 1
  + An image with aspect ratio [1, 2] would have ID 2
  + An image with aspect ratio [2, 1] would have ID 3

  The id 0 is reserved for padding (i.e., no image).

  If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **cross\_attention\_mask** (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*) —
  Cross-attention mask to control the interaction between text tokens and image tiles.
  This 4D tensor defines which image tiles each text token should attend to.

  For each text token (in seq\_length):

  + 1 indicates the token **should attend** to the corresponding image tile
  + 0 indicates the token **should not attend** to the corresponding image tile
* **cross\_attention\_states** (`torch.FloatTensor`, *optional*) —
  Output of the vision model, used for cross-attention. This tensor contains the processed image features that
  the language model will attend to.
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MllamaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MllamaForConditionalGeneration

>>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
>>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompt = "<|image|>If I had to write a haiku for this one"
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")

>>> # Generate
>>> output = model.generate(**inputs, max_new_tokens=15)

>>> prompt_len = inputs.input_ids.shape[-1]
>>> generated_ids = output[:, prompt_len:]
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
>>> print(generated_text)
[', it would be:.\\nA stop sign in Chinatown.\\n']
```

## MllamaForCausalLM

### class transformers.MllamaForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1287)

( config  )

Parameters

* **config** ([MllamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mllama Text Model with a language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1302)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None cross\_attention\_states: typing.Optional[torch.LongTensor] = None cross\_attention\_mask: typing.Optional[torch.LongTensor] = None full\_text\_row\_masked\_out\_mask: typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **cross\_attention\_states** (`torch.FloatTensor`, *optional*) —
  Output of the vision model, used for cross-attention. This tensor contains the processed image features that
  the language model will attend to.
* **cross\_attention\_mask** (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*) —
  Cross-attention mask to control the interaction between text tokens and image tiles.
  This 4D tensor defines which image tiles each text token should attend to.

  For each text token (in seq\_length):

  + 1 indicates the token **should attend** to the corresponding image tile
  + 0 indicates the token **should not attend** to the corresponding image tile
* **full\_text\_row\_masked\_out\_mask** (`tuple[torch.Tensor, torch.Tensor]`, *optional*) —
  A tuple containing two tensors that mask out rows in the cross-attention mechanism:
  + The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
    A value of 0 indicates that the corresponding text token’s entire row in the cross-attention
    matrix should be masked out (all image tokens ignored).
  + The second tensor has the same shape and is used internally to apply the masking during
    the forward pass of cross-attention layers.
    This mask is derived from the cross\_attention\_mask and is used to handle cases where a text token
    should not attend to any image token.
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MllamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MllamaForCausalLM

>>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
>>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

>>> prompt = "If I had to write a haiku, it would be:"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
>>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
>>> print(result)
If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
I love the idea of snowflakes gently falling, each one
```

## MllamaTextModel

### class transformers.MllamaTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1141)

( config: MllamaTextConfig  )

Parameters

* **config** (`MllamaTextConfig`) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mllama Text Model which consists of transformer with self and cross attention layers.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1165)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None cross\_attention\_states: typing.Optional[torch.FloatTensor] = None cross\_attention\_mask: typing.Optional[torch.Tensor] = None full\_text\_row\_masked\_out\_mask: typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **cross\_attention\_states** (`torch.FloatTensor`, *optional*) —
  Output of the vision model, used for cross-attention. This tensor contains the processed image features that
  the language model will attend to.
* **cross\_attention\_mask** (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*) —
  Cross-attention mask to control the interaction between text tokens and image tiles.
  This 4D tensor defines which image tiles each text token should attend to.

  For each text token (in seq\_length):

  + 1 indicates the token **should attend** to the corresponding image tile
  + 0 indicates the token **should not attend** to the corresponding image tile
* **full\_text\_row\_masked\_out\_mask** (`tuple[torch.Tensor, torch.Tensor]`, *optional*) —
  A tuple containing two tensors that mask out rows in the cross-attention mechanism:
  + The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
    A value of 0 indicates that the corresponding text token’s entire row in the cross-attention
    matrix should be masked out (all image tokens ignored).
  + The second tensor has the same shape and is used internally to apply the masking during
    the forward pass of cross-attention layers.
    This mask is derived from the cross\_attention\_mask and is used to handle cases where a text token
    should not attend to any image token.
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MllamaTextModel](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, MllamaTextModel

>>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
>>> model = MllamaTextModel.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> text = "<|image|>If I had to write a haiku for this one"
>>> inputs = processor(text=text, return_tensors="pt")

>>> output = model(**inputs)

>>> print(output.last_hidden_state.shape)
torch.Size([1, 13, 4096])
```

## MllamaModel

### class transformers.MllamaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1401)

( config: MllamaConfig  )

Parameters

* **config** ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mllama model which consists of a vision encoder and a language model without language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1433)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None aspect\_ratio\_mask: typing.Optional[torch.Tensor] = None aspect\_ratio\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None cross\_attention\_mask: typing.Optional[torch.Tensor] = None cross\_attention\_states: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor). See [MllamaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MllamaProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor) uses
  [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor) for processing images).
* **aspect\_ratio\_mask** (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*) —
  Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:
  + 1 for tiles that are **not masked**,
  + 0 for tiles that are **masked**.
* **aspect\_ratio\_ids** (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*) —
  Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
  These ids correspond to indices in the model’s list of supported aspect ratios, offset by 1.

  For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:

  + An image with aspect ratio [1, 1] would have ID 1
  + An image with aspect ratio [1, 2] would have ID 2
  + An image with aspect ratio [2, 1] would have ID 3

  The id 0 is reserved for padding (i.e., no image).

  If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **cross\_attention\_mask** (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*) —
  Cross-attention mask to control the interaction between text tokens and image tiles.
  This 4D tensor defines which image tiles each text token should attend to.

  For each text token (in seq\_length):

  + 1 indicates the token **should attend** to the corresponding image tile
  + 0 indicates the token **should not attend** to the corresponding image tile
* **cross\_attention\_states** (`torch.FloatTensor`, *optional*) —
  Output of the vision model, used for cross-attention. This tensor contains the processed image features that
  the language model will attend to.
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MllamaModel](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MllamaForCausalLM

### class transformers.MllamaForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1287)

( config  )

Parameters

* **config** ([MllamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mllama Text Model with a language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1302)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None cross\_attention\_states: typing.Optional[torch.LongTensor] = None cross\_attention\_mask: typing.Optional[torch.LongTensor] = None full\_text\_row\_masked\_out\_mask: typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **cross\_attention\_states** (`torch.FloatTensor`, *optional*) —
  Output of the vision model, used for cross-attention. This tensor contains the processed image features that
  the language model will attend to.
* **cross\_attention\_mask** (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*) —
  Cross-attention mask to control the interaction between text tokens and image tiles.
  This 4D tensor defines which image tiles each text token should attend to.

  For each text token (in seq\_length):

  + 1 indicates the token **should attend** to the corresponding image tile
  + 0 indicates the token **should not attend** to the corresponding image tile
* **full\_text\_row\_masked\_out\_mask** (`tuple[torch.Tensor, torch.Tensor]`, *optional*) —
  A tuple containing two tensors that mask out rows in the cross-attention mechanism:
  + The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
    A value of 0 indicates that the corresponding text token’s entire row in the cross-attention
    matrix should be masked out (all image tokens ignored).
  + The second tensor has the same shape and is used internally to apply the masking during
    the forward pass of cross-attention layers.
    This mask is derived from the cross\_attention\_mask and is used to handle cases where a text token
    should not attend to any image token.
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MllamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MllamaForCausalLM

>>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
>>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

>>> prompt = "If I had to write a haiku, it would be:"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
>>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
>>> print(result)
If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
I love the idea of snowflakes gently falling, each one
```

## MllamaVisionModel

### class transformers.MllamaVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L944)

( config: MllamaVisionConfig  )

Parameters

* **config** (`MllamaVisionConfig`) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mllama Vision Model which consists of two vision encoders.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L997)

( pixel\_values: Tensor aspect\_ratio\_ids: Tensor aspect\_ratio\_mask: Tensor \*\*kwargs  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor). See [MllamaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MllamaProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor) uses
  [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor) for processing images).
* **aspect\_ratio\_ids** (`torch.Tensor` of shape `(batch_size, max_num_images)`, *optional*) —
  Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
  These ids correspond to indices in the model’s list of supported aspect ratios, offset by 1.

  For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:

  + An image with aspect ratio [1, 1] would have ID 1
  + An image with aspect ratio [1, 2] would have ID 2
  + An image with aspect ratio [2, 1] would have ID 3

  The id 0 is reserved for padding (i.e., no image).

  If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its `aspect_ratio_id` would be 2.
* **aspect\_ratio\_mask** (`torch.Tensor` of shape `(batch_size, max_num_images, max_num_tiles)`, *optional*) —
  Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:
  + 1 for tiles that are **not masked**,
  + 0 for tiles that are **masked**.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MllamaVisionModel](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MllamaVisionModel

>>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
>>> model = MllamaVisionModel.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")

>>> output = model(**inputs)

>>> print(output.last_hidden_state.shape)
torch.Size([1, 1, 4, 1025, 7680])
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mllama.md)
