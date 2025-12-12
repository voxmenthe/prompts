*This model was released on 2025-02-19 and added to Hugging Face Transformers on 2025-01-23.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Qwen2.5-VL

[Qwen2.5-VL](https://huggingface.co/papers/2502.13923) is a multimodal vision-language model, available in 3B, 7B, and 72B parameters, pretrained on 4.1T tokens. The model introduces window attention in the ViT encoder to accelerate training and inference, dynamic FPS sampling on the spatial and temporal dimensions for better video understanding across different sampling rates, and an upgraded MRoPE (multi-resolutional rotary positional encoding) mechanism to better capture and learn temporal dynamics.

You can find all the original Qwen2.5-VL checkpoints under the [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) collection.

Click on the Qwen2.5-VL models in the right sidebar for more examples of how to apply Qwen2.5-VL to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline
pipe = pipeline(
    task="image-text-to-text",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    device=0,
    dtype=torch.bfloat16
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
pipe(text=messages,max_new_tokens=20, return_full_text=False)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
import torch
from transformers import TorchAoConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
```

### Notes

* Use Qwen2.5-VL for video inputs by setting `"type": "video"` as shown below.


  ```
  conversation = [
      {
          "role": "user",
          "content": [
              {"type": "video", "path": "/path/to/video.mp4"},
              {"type": "text", "text": "What happened in the video?"},
          ],
      }
  ]

  inputs = processor.apply_chat_template(
      conversation,
      video_fps=1,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt"
  ).to(model.device)

  # Inference: Generation of the output
  output_ids = model.generate(**inputs, max_new_tokens=128)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  print(output_text)
  ```
* Use Qwen2.5-VL for a mixed batch of inputs (images, videos, text). Add labels when handling multiple images or videos for better reference
  as show below.


  ```
  import torch
  from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      "Qwen/Qwen2.5-VL-7B-Instruct",
      dtype=torch.float16,
      device_map="auto",
      attn_implementation="sdpa"
  )
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
  conversation = [
      {
          "role": "user",
          "content": [
              {"type": "image"}, 
              {"type": "text", "text": "Hello, how are you?"}
          ]
      },
      {
          "role": "assistant",
          "content": "I'm doing well, thank you for asking. How can I assist you today?"
      },
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Can you describe these images and video?"}, 
              {"type": "image"}, 
              {"type": "image"}, 
              {"type": "video"}, 
              {"type": "text", "text": "These are from my vacation."}
          ]
      },
      {
          "role": "assistant",
          "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
      },
      {
          "role": "user",
          "content": "It was a trip to the mountains. Can you see the details in the images and video?"
      }
  ]

  # default:
  prompt_without_id = processor.apply_chat_template(conversation, add_generation_prompt=True)
  # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'


  # add ids
  prompt_with_id = processor.apply_chat_template(conversation, add_generation_prompt=True, add_vision_id=True)
  # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?Picture 2: <|vision_start|><|image_pad|><|vision_end|>Picture 3: <|vision_start|><|image_pad|><|vision_end|>Video 1: <|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
  ```
* Use the `min_pixels` and `max_pixels` parameters in [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) to set the resolution.


  ```
  min_pixels = 224*224
  max_pixels = 2048*2048
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
  ```

  Higher resolution can require more compute whereas reducing the resolution can save memory as follows:


  ```
  min_pixels = 256*28*28
  max_pixels = 1024*28*28 
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
  ```

## Qwen2\_5\_VLConfig

### class transformers.Qwen2\_5\_VLConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py#L273)

( text\_config = None vision\_config = None image\_token\_id = 151655 video\_token\_id = 151656 \*\*kwargs  )

Parameters

* **text\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLTextConfig`) —
  The config object or dictionary of the text backbone.
* **vision\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 151655) —
  The image token index to encode the image prompt.
* **video\_token\_id** (`int`, *optional*, defaults to 151656) —
  The video token index to encode the image prompt.

This is the configuration class to store the configuration of a [Qwen2\_5\_VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLModel). It is used to instantiate a
Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

>>> # Initializing a Qwen2_5_VL style configuration
>>> configuration = Qwen2_5_VLConfig()

>>> # Initializing a model from the Qwen2-VL-7B style configuration
>>> model = Qwen2_5_VLForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2\_5\_VLTextConfig

### class transformers.Qwen2\_5\_VLTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py#L70)

( vocab\_size = 152064 hidden\_size = 8192 intermediate\_size = 29568 num\_hidden\_layers = 80 num\_attention\_heads = 64 num\_key\_value\_heads = 8 hidden\_act = 'silu' max\_position\_embeddings = 32768 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True tie\_word\_embeddings = False rope\_theta = 1000000.0 use\_sliding\_window = False sliding\_window = 4096 max\_window\_layers = 80 layer\_types = None attention\_dropout = 0.0 rope\_scaling = None image\_token\_id = None video\_token\_id = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 152064) —
  Vocabulary size of the Qwen2\_5\_VL model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Qwen2\_5\_VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLModel)
* **hidden\_size** (`int`, *optional*, defaults to 8192) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 29568) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 80) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 64) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 32768) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **rope\_theta** (`float`, *optional*, defaults to 1000000.0) —
  The base period of the RoPE embeddings.
* **use\_sliding\_window** (`bool`, *optional*, defaults to `False`) —
  Whether to use sliding window attention.
* **sliding\_window** (`int`, *optional*, defaults to 4096) —
  Sliding window attention (SWA) window size. If not specified, will default to `4096`.
* **max\_window\_layers** (`int`, *optional*, defaults to 80) —
  The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
  additional layer afterwards will use SWA (Sliding Window Attention).
* **layer\_types** (`list`, *optional*) —
  Attention pattern for each layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
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
* **image\_token\_id** (`int`, *optional*) —
  Token index used as placeholder for image embeddings.
* **video\_token\_id** (`int`, *optional*) —
  Token index used as placeholder for video embeddings.

This is the configuration class to store the configuration of a [Qwen2\_5\_VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextModel). It is used to instantiate a
Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Qwen2_5_VLTextModel, Qwen2_5_VLConfig

>>> # Initializing a Qwen2_5_VL style configuration
>>> configuration = Qwen2_5_VLConfig()

>>> # Initializing a model from the Qwen2-VL-7B style configuration
>>> model = Qwen2_5_VLTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2\_5\_VLProcessor

### class transformers.Qwen2\_5\_VLProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py#L61)

( image\_processor = None tokenizer = None video\_processor = None chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast), *optional*) —
  The tokenizer is a required input.
* **video\_processor** (`Qwen2_5_VLVideoProcessor`, *optional*) —
  The video processor is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a Qwen2.5-VL processor which wraps a Qwen2.5-VL image processor and a Qwen2 tokenizer into a single processor.
[Qwen2\_5\_VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) and [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### post\_process\_image\_text\_to\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py#L243)

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

## Qwen2\_5\_VLTextModel

### class transformers.Qwen2\_5\_VLTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L781)

( config: Qwen2\_5\_VLTextConfig  )

Parameters

* **config** ([Qwen2\_5\_VLTextConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen2 5 Vl Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L802)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5\_VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig)) and inputs.

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

The [Qwen2\_5\_VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen2\_5\_VLModel

### class transformers.Qwen2\_5\_VLModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L938)

( config  )

Parameters

* **config** ([Qwen2\_5\_VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen2 5 Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1223)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None second\_per\_grid\_ts: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Qwen2\_5\_VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor) uses
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor). See `Qwen2VLVideoProcessor.__call__()` for details ([Qwen2\_5\_VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor) uses
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **second\_per\_grid\_ts** (`torch.Tensor` of shape `(num_videos)`, *optional*) —
  The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

Returns

`transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5\_VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig)) and inputs.

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Qwen2\_5\_VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen2\_5\_VLForConditionalGeneration

### class transformers.Qwen2\_5\_VLForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1367)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1412)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None second\_per\_grid\_ts: typing.Optional[torch.Tensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Qwen2\_5\_VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor) uses
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor). See `Qwen2VLVideoProcessor.__call__()` for details ([Qwen2\_5\_VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor) uses
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **second\_per\_grid\_ts** (`torch.Tensor` of shape `(num_videos)`, *optional*) —
  The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5\_VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig)) and inputs.

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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Qwen2\_5\_VLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

>>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

>>> messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
>>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_vl.md)
