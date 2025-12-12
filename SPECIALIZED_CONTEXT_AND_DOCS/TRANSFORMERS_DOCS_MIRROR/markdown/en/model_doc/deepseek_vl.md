*This model was released on 2024-03-08 and added to Hugging Face Transformers on 2025-07-25.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# DeepseekVL

[Deepseek-VL](https://huggingface.co/papers/2403.05525) was introduced by the DeepSeek AI team. It is a vision-language model (VLM) designed to process both text and images for generating contextually relevant responses. The model leverages [LLaMA](./llama) as its text encoder, while [SigLip](./siglip) is used for encoding images.

You can find all the original Deepseek-VL checkpoints under the [DeepSeek-community](https://huggingface.co/deepseek-community) organization.

Click on the Deepseek-VL models in the right sidebar for more examples of how to apply Deepseek-VL to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="deepseek-community/deepseek-vl-1.3b-chat",
    device=0,
    dtype=torch.float16
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe this image."},
        ]
    }
]

pipe(text=messages, max_new_tokens=20, return_full_text=False)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
import torch
from transformers import TorchAoConfig, DeepseekVLForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig(
    "int4_weight_only",
    group_size=128
)

model = DeepseekVLForConditionalGeneration.from_pretrained(
    "deepseek-community/deepseek-vl-1.3b-chat",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
```

### Notes

* Do inference with multiple images in a single conversation.


  ```
  import torch
  from transformers import DeepseekVLForConditionalGeneration, AutoProcessor

  model = DeepseekVLForConditionalGeneration.from_pretrained(
      "deepseek-community/deepseek-vl-1.3b-chat",
      dtype=torch.float16,
      device_map="auto",
      attn_implementation="sdpa"
  )

  processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat")

  messages = [
      [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "What’s the difference between"},
                  {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                  {"type": "text", "text": " and "},
                  {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
              ]
          }
      ],
      [
          {
              "role": "user",
              "content": [
                  {"type": "image", "url": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"},
                  {"type": "text", "text": "What do you see in this image?"}
              ]
          }
      ]
  ]

  inputs = processor.apply_chat_template(
      messages,
      add_generation_prompt=True,
      padding=True,
      truncation=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt"
  ).to(model.device, dtype=model.dtype)

  generated_ids = model.generate(**inputs, max_new_tokens=128)
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
  )

  print(output_text)
  ```

## DeepseekVLConfig

### class transformers.DeepseekVLConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/configuration_deepseek_vl.py#L32)

( text\_config: AutoConfig = None vision\_config: AutoConfig = None image\_token\_id: int = 100015 \*\*kwargs  )

Parameters

* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`) —
  The config object or dictionary of the text backbone.
* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `SiglipVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 100015) —
  The index representing image tokens in the model’s token vocabulary.

This is the configuration class to store the configuration of a [DeepseekVLModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLModel). It is used to instantiate a
DeepseekVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the DeepseekVL
[deepseek-community/deepseek-vl-1.3b-chat](https://huggingface.co/deepseek-community/deepseek-vl-1.3b-chat) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DeepseekVLConfig, DeepseekVLModel

>>> # Initializing a DeepseekVL deepseek-community/deepseek-vl-1.3b-chat style configuration
>>> configuration = DeepseekVLConfig()

>>> # Initializing a model (with random weights) from the deepseek-community/deepseek-vl-1.3b-chat style configuration
>>> model = DeepseekVLModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DeepseekVLProcessor

### class transformers.DeepseekVLProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/processing_deepseek_vl.py#L36)

( image\_processor tokenizer chat\_template = None num\_image\_tokens = 576  )

Parameters

* **image\_processor** ([DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor)) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast)) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) —
  A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.
* **num\_image\_tokens** (`int`, *optional*, defaults to 576) —
  The number of special image tokens used as placeholders for visual content in text sequences.

Constructs a DeepseekVL processor which wraps a DeepseekVL Image Processor and a Llama tokenizer into a single processor.

[DeepseekVLProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLProcessor) offers all the functionalities of [DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLProcessor.decode) for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/processing_deepseek_vl.py#L135)

( \*args \*\*kwargs  )

This method forwards all its arguments to LlamaTokenizerFast’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/processing_deepseek_vl.py#L142)

( \*args \*\*kwargs  )

This method forwards all its arguments to LlamaTokenizerFast’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## DeepseekVLImageProcessor

### class transformers.DeepseekVLImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/image_processing_deepseek_vl.py#L56)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None min\_size: int = 14 resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"height" -- 384, "width": 384}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **min\_size** (`int`, *optional*, defaults to 14) —
  The minimum allowed size for the resized image. Ensures that neither the height nor width
  falls below this value after resizing.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a DEEPSEEK\_VL image processor.

#### pad\_to\_square

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/image_processing_deepseek_vl.py#L342)

( image: ndarray background\_color: typing.Union[int, tuple[int, int, int]] = 0 data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  The image to pad.
* **background\_color** (`int` or `tuple[int, int, int]`, *optional*, defaults to 0) —
  The color to use for the padding. Can be an integer for single channel or a
  tuple of integers representing for multi-channel images. If passed as integer
  in mutli-channel mode, it will default to `0` in subsequent channels.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use same as the input image.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.

Returns

`np.ndarray`

The padded image.

Pads an image to a square based on the longest edge.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/image_processing_deepseek_vl.py#L205)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Controls the size of the image after `resize`. The shortest edge of the image is resized to
  `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
  is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
  edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to normalize the image by if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
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

Preprocess an image or batch of images.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/image_processing_deepseek_vl.py#L130)

( image: ndarray size: typing.Union[dict[str, int], int] background\_color: typing.Optional[tuple[int, int, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]` or `int`) —
  The size to resize the image to. If a dictionary, it should have the keys `"height"` and `"width"`.
* **background\_color** (`tuple[int, int, int]`) —
  The background color to use for the padding.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
* **data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `None`: will be inferred from input
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Returns

`np.ndarray`

The resized image.

Resize an image to dynamically calculated size.

## DeepseekVLImageProcessorFast

### class transformers.DeepseekVLImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/image_processing_deepseek_vl_fast.py#L56)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.deepseek\_vl.image\_processing\_deepseek\_vl\_fast.DeepseekVLFastImageProcessorKwargs]  )

Constructs a fast Deepseek Vl image processor.

#### pad\_to\_square

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/image_processing_deepseek_vl_fast.py#L102)

( images: torch.Tensor background\_color: typing.Union[int, tuple[int, int, int]] = 0  ) → `torch.Tensor`

Parameters

* **images** (`torch.Tensor`) —
  The images to pad.
* **background\_color** (`int` or `tuple[int, int, int]`, *optional*, defaults to 0) —
  The color to use for the padding. Can be an integer for single channel or a
  tuple of integers representing for multi-channel images. If passed as integer
  in mutli-channel mode, it will default to `0` in subsequent channels.

Returns

`torch.Tensor`

The padded images.

Pads an image to a square based on the longest edge.

## DeepseekVLModel

### class transformers.DeepseekVLModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/modeling_deepseek_vl.py#L155)

( config  )

Parameters

* **config** ([DeepseekVLModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Deepseek Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/modeling_deepseek_vl.py#L204)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor). See [DeepseekVLImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([DeepseekVLProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLProcessor) uses
  [DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

The [DeepseekVLModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## DeepseekVLForConditionalGeneration

### class transformers.DeepseekVLForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/modeling_deepseek_vl.py#L255)

( config: DeepseekVLConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_vl/modeling_deepseek_vl.py#L277)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor). See [DeepseekVLImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([DeepseekVLProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLProcessor) uses
  [DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
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
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

The [DeepseekVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, DeepseekVLForConditionalGeneration

>>> model = DeepseekVLForConditionalGeneration.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat")
>>> processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat")

>>> messages = [
...     {
...         "role": "user", "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "Where is the cat standing?"},
...         ]
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     add_generation_prompt=True
... )
>>> # Generate
>>> generate_ids = model.generate(**inputs)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_vl.md)
