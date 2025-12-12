*This model was released on 2024-10-08 and added to Hugging Face Transformers on 2024-12-06.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Aria

[Aria](https://huggingface.co/papers/2410.05993) is a multimodal mixture-of-experts (MoE) model. The goal of this model is to open-source a training recipe for creating a multimodal native model from scratch. Aria has 3.9B and 3.5B activated parameters per visual and text token respectively. Text is handled by a MoE decoder and visual inputs are handled by a lightweight visual encoder. It is trained in 4 stages, language pretraining, multimodal pretraining, multimodal long-context pretraining, and multimodal post-training.

You can find all the original Aria checkpoints under the [Aria](https://huggingface.co/rhymes-ai?search_models=aria) organization.

Click on the Aria models in the right sidebar for more examples of how to apply Aria to different multimodal tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    "image-to-text",
    model="rhymes-ai/Aria",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="What is shown in this image?"
)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4 and the [rhymes-ai/Aria-sequential\_mlp](https://huggingface.co/rhymes-ai/Aria-sequential_mlp) checkpoint. This checkpoint replaces grouped GEMM with `torch.nn.Linear` layers for easier quantization.


```
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoProcessor

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "rhymes-ai/Aria-sequential_mlp",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    "rhymes-ai/Aria-sequential_mlp",
)

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device, torch.bfloat16)

output = model.generate(
    **inputs,
    max_new_tokens=15,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(output_ids, skip_special_tokens=True)
print(response)
```

## AriaImageProcessor

### class transformers.AriaImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L74)

( image\_mean: typing.Optional[list[float]] = None image\_std: typing.Optional[list[float]] = None max\_image\_size: int = 980 min\_image\_size: int = 336 split\_resolutions: typing.Optional[list[tuple[int, int]]] = None split\_image: typing.Optional[bool] = False do\_convert\_rgb: typing.Optional[bool] = True do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: typing.Optional[bool] = True resample: Resampling = <Resampling.BICUBIC: 3> \*\*kwargs  )

Parameters

* **image\_mean** (`list`, *optional*, defaults to [0.5, 0.5, 0.5]) —
  Mean values for normalization.
* **image\_std** (`list`, *optional*, defaults to [0.5, 0.5, 0.5]) —
  Standard deviation values for normalization.
* **max\_image\_size** (`int`, *optional*, defaults to 980) —
  Maximum image size.
* **min\_image\_size** (`int`, *optional*, defaults to 336) —
  Minimum image size.
* **split\_resolutions** (`list`, *optional*, defaults to a list of optimal,resolutions as tuples) —
  The optimal resolutions for splitting the image.
* **split\_image** (`bool`, *optional*, defaults to `False`) —
  Whether to split the image.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image.
* **resample** (PILImageResampling, *optional*, defaults to `BICUBIC`) —
  The resampling filter to use if resizing the image.

A vision processor for the Aria model that handles image preprocessing.
Initialize the AriaImageProcessor.

#### get\_image\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L455)

( image: <built-in function array> grid\_pinpoints: list patch\_size: int resample: Resampling data\_format: ChannelDimension input\_data\_format: ChannelDimension  ) → `list[np.array]`

Parameters

* **image** (`np.array`) —
  The input image to be processed.
* **grid\_pinpoints** (list[tuple[int, int]]) —
  A list of possible resolutions as tuples.
* **patch\_size** (`int`) —
  Size of the patches to divide the image into.
* **resample** (`PILImageResampling`) —
  Resampling filter to use if resizing the image.
* **data\_format** (`ChannelDimension` or `str`) —
  The channel dimension format for the output image.
* **input\_data\_format** (`ChannelDimension` or `str`) —
  The channel dimension format of the input image.

Returns

`list[np.array]`

A list of NumPy arrays containing the processed image patches.

Process an image with variable resolutions by dividing it into patches.

#### get\_number\_of\_image\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L505)

( height: int width: int images\_kwargs = None  ) → `int`

Parameters

* **height** (`int`) —
  Height of the input image.
* **width** (`int`) —
  Width of the input image.
* **images\_kwargs** (`dict`, *optional*) —
  Any kwargs to override defaults of the image processor.

Returns

`int`

Number of patches per image.

A utility that returns number of image patches for a given image size.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L389)

( image: ndarray padding: typing.Union[int, tuple[int, int], collections.abc.Iterable[tuple[int, int]]] mode: PaddingMode = <PaddingMode.CONSTANT: 'constant'> constant\_values: typing.Union[float, collections.abc.Iterable[float]] = 0.0 data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  The image to pad.
* **padding** (`int` or `tuple[int, int]` or `Iterable[tuple[int, int]]`) —
  Padding to apply to the edges of the height, width axes. Can be one of three formats:
  + `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
  + `((before, after),)` yields same before and after pad for height and width.
  + `(pad,)` or int is a shortcut for before = after = pad width for all axes.
* **mode** (`PaddingMode`) —
  The padding mode to use. Can be one of:
  + `"constant"`: pads with a constant value.
  + `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
    vector along each axis.
  + `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
  + `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
* **constant\_values** (`float` or `Iterable[float]`, *optional*) —
  The value to use for the padding if `mode` is `"constant"`.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use same as the input image.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use the inferred format of the input image.

Returns

`np.ndarray`

The padded image.

Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
as input.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L144)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]] image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None max\_image\_size: typing.Optional[int] = None min\_image\_size: typing.Optional[int] = None split\_image: typing.Optional[bool] = None do\_convert\_rgb: typing.Optional[bool] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None resample: Resampling = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = 'pt' data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  ) → BatchFeature

Parameters

* **images** (ImageInput or list of ImageInput) —
  The input image or a list of images.
* **image\_mean** (`list`, *optional*, defaults to [0.5, 0.5, 0.5]) —
  Mean values for normalization.
* **image\_std** (`list`, *optional*, defaults to [0.5, 0.5, 0.5]) —
  Standard deviation values for normalization.
* **max\_image\_size** (`int`, *optional*, defaults to `self.max_image_size` (980)) —
  Maximum image size.
* **min\_image\_size** (`int`, *optional*, defaults to `self.min_image_size` (336)) —
  Minimum image size.
* **split\_image** (`bool`, *optional*, defaults to `self.split_image` (False)) —
  Whether to split the image.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb` (True)) —
  Whether to convert the image to RGB.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize` (True)) —
  Whether to normalize the image.
* **resample** (PILImageResampling, *optional*, defaults to `self.resample` (BICUBIC)) —
  The resampling filter to use if resizing the image.
* **return\_tensors** (`str` or `TensorType`, *optional*, defaults to “pt”) —
  The type of tensor to return.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`:
    image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`:
    image in (height, width, num\_channels) format.
    If unset, will use same as the input image.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`:
    image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`:
    image in (height, width, num\_channels) format.
    If unset, will use the inferred format of the input image.

Returns

BatchFeature

A BatchFeature object containing:

* ‘pixel\_values’:
  Tensor of processed image pixel values.
* ‘pixel\_mask’:
  Boolean pixel mask. This mask is a 2D tensor of shape (max\_image\_size, max\_image\_size) where:
  + True (1) values indicate pixels that belong to the original resized image.
  + False (0) values indicate pixels that are part of the padding.
    The mask helps distinguish between actual image content and padded areas in subsequent processing steps.
* ‘num\_crops’:
  The maximum number of crops across all images.

Process a list of images.

## AriaProcessor

### class transformers.AriaProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/processing_aria.py#L47)

( image\_processor = None tokenizer: typing.Union[transformers.models.auto.tokenization\_auto.AutoTokenizer, str] = None chat\_template: typing.Optional[str] = None size\_conversion: typing.Optional[dict[typing.Union[float, int], int]] = None  )

Parameters

* **image\_processor** (`AriaImageProcessor`, *optional*) —
  The AriaImageProcessor to use for image preprocessing.
* **tokenizer** (`PreTrainedTokenizerBase`, *optional*) —
  An instance of [PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase). This should correspond with the model’s text model. The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) —
  A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
* **size\_conversion** (`Dict`, *optional*) —
  A dictionary indicating size conversions for images.

AriaProcessor is a processor for the Aria model which wraps the Aria image preprocessor and the LLama slow tokenizer.

## AriaTextConfig

### class transformers.AriaTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/configuration_aria.py#L28)

( vocab\_size = 32000 hidden\_size = 4096 intermediate\_size: int = 4096 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = None hidden\_act = 'silu' max\_position\_embeddings = 2048 initializer\_range = 0.02 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = 2 bos\_token\_id = 1 eos\_token\_id = 2 pretraining\_tp = 1 tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None attention\_bias = False attention\_dropout = 0.0 mlp\_bias = False head\_dim = None moe\_num\_experts: int = 8 moe\_topk: int = 2 moe\_num\_shared\_experts: int = 2 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [LlamaModel](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 4096) —
  The size of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 2048) —
  The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
  Llama 2 up to 4096, CodeLlama up to 16384.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 2) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  End of stream token id.
* **pretraining\_tp** (`int`, *optional*, defaults to 1) —
  Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
  document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
  understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
  results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *original maximum pre-trained length.
  `original_max_position_embeddings` (`int`,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`,* optional*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
  `beta_fast` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.
  `beta_slow` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.
  `short_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in up\_proj, down\_proj and gate\_proj layers in the MLP layers.
* **head\_dim** (`int`, *optional*) —
  The attention head dimension. If None, it will default to hidden\_size // num\_heads
* **moe\_num\_experts** (`int`, *optional*, defaults to 8) —
  The number of experts in the MoE layer.
* **moe\_topk** (`int`, *optional*, defaults to 2) —
  The number of top experts to route to for each token.
* **moe\_num\_shared\_experts** (`int`, *optional*, defaults to 2) —
  The number of shared experts.

This class handles the configuration for the text component of the Aria model.
Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the Aria
[rhymes-ai/Aria](https://huggingface.co/rhymes-ai/Aria) architecture.
This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.

## AriaConfig

### class transformers.AriaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/configuration_aria.py#L223)

( vision\_config = None vision\_feature\_layer: int = -1 text\_config: AriaTextConfig = None projector\_patch\_to\_query\_dict: typing.Optional[dict] = None image\_token\_index: int = 9 initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **vision\_config** (`AriaVisionConfig` or `dict`, *optional*) —
  Configuration for the vision component.
* **vision\_feature\_layer** (`int`, *optional*, defaults to -1) —
  The index of the layer to select the vision feature.
* **text\_config** (`AriaTextConfig` or `dict`, *optional*) —
  Configuration for the text component.
* **projector\_patch\_to\_query\_dict** (`dict`, *optional*) —
  Mapping of patch sizes to query dimensions.
* **image\_token\_index** (`int`, *optional*, defaults to 9) —
  Index used to represent image tokens.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated normal initializer for initializing all weight matrices.
* **model\_type** (`str`) —
  Type of the model, set to `"aria"`.
* **image\_token\_index** (`int`) —
  Index used to represent image tokens.
* **projector\_patch\_to\_query\_dict** (`dict`) —
  Mapping of patch sizes to query dimensions.
* **vision\_config** (`AriaVisionConfig`) —
  Configuration for the vision component.
* **text\_config** (`AriaTextConfig`) —
  Configuration for the text component.

This class handles the configuration for both vision and text components of the Aria model,
as well as additional parameters for image token handling and projector mapping.
Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the Aria
[rhymes-ai/Aria](https://huggingface.co/rhymes-ai/Aria) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## AriaTextModel

### class transformers.AriaTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L710)

( config: AriaTextConfig  )

Parameters

* **config** ([AriaTextConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Aria Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L727)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig)) and inputs.

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

The [AriaTextModel](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## AriaModel

### class transformers.AriaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L921)

( config: AriaConfig  )

Parameters

* **config** ([AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Aria model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L1004)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None pixel\_mask: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.aria.modeling_aria.AriaModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [AriaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor). See [AriaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AriaProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaProcessor) uses
  [AriaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor) for processing images).
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
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

`transformers.models.aria.modeling_aria.AriaModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.aria.modeling_aria.AriaModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [AriaModel](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L943)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: int = -1  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) —
  The tensors corresponding to the input images.
* **pixel\_mask** (`torch.FloatTensor]`, *optional*) —
  The tensors corresponding to the input image mask.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_images, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L980)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## AriaTextForCausalLM

### class transformers.AriaTextForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L789)

( config: AriaTextConfig  )

Parameters

* **config** ([AriaTextConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Aria Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L803)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

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
elements depending on the configuration ([AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig)) and inputs.

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

The [AriaTextForCausalLM](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, AriaTextForCausalLM

>>> model = AriaTextForCausalLM.from_pretrained("meta-aria_text/AriaText-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-aria_text/AriaText-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## AriaForConditionalGeneration

### class transformers.AriaForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L1078)

( config: AriaConfig  )

Parameters

* **config** ([AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Aria model for conditional generation tasks.

This model combines a vision tower, a multi-modal projector, and a language model
to perform tasks that involve both image and text inputs.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L1133)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None pixel\_mask: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.aria.modeling_aria.AriaCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [AriaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor). See [AriaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AriaProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaProcessor) uses
  [AriaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor) for processing images).
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
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
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `AriaForConditionalGeneration`).
  Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
  computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.aria.modeling_aria.AriaCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.aria.modeling_aria.AriaCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [AriaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from io import BytesIO

>>> from transformers import AutoProcessor, AutoModel
>>> from transformers.image_utils import load_image

>>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
>>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
>>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
>>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

>>> processor = AutoProcessor.from_pretrained("Rhymes-AI/Aria")
>>> model = AutoModel.from_pretrained("Rhymes-AI/Aria", dtype=torch.bfloat16, device_map="auto")

>>> # Create inputs
>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image"},
...             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
...             {"type": "image"},
...             {"type": "text", "text": "What can we see in this image?"},
...         ]
...     },
...     {
...         "role": "user",
...         "content": [
...             {"type": "image"},
...             {"type": "text", "text": "In which city is that bridge located?"},
...         ]
...     }
... ]

>>> prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
>>> images = [[image1, image2], [image3]]
>>> inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

>>> # Generate
>>> generated_ids = model.generate(**inputs, max_new_tokens=256)
>>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

>>> print(generated_texts[0])
Assistant: There are buildings, trees, lights, and water visible in this image.

>>> print(generated_texts[1])
Assistant: The bridge is in San Francisco.
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/aria.md)
