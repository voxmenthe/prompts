*This model was released on 2024-09-17 and added to Hugging Face Transformers on 2024-09-18.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Mimi

[Mimi](huggingface.co/papers/2410.00037) is a neural audio codec model with pretrained and quantized variants, designed for efficient speech representation and compression. The model operates at 1.1 kbps with a 12 Hz frame rate and uses a convolutional encoder-decoder architecture combined with a residual vector quantizer of 16 codebooks. Mimi outputs dual token streams i.e. semantic and acoustic to balance linguistic richness with high fidelity reconstruction. Key features include a causal streaming encoder for low-latency use, dual-path tokenization for flexible downstream generation, and integration readiness with large speech models like Moshi.

You can find the original Mimi checkpoints under the [Kyutai](https://huggingface.co/kyutai/models?search=mimi) organization.

This model was contributed by [ylacombe](https://huggingface.co/ylacombe).

Click on the Mimi models in the right sidebar for more examples of how to apply Mimi.

The example below demonstrates how to encode and decode audio with the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

AutoModel


```
>>> from datasets import load_dataset, Audio
>>> from transformers import MimiModel, AutoFeatureExtractor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> # load model and feature extractor
>>> model = MimiModel.from_pretrained("kyutai/mimi")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

>>> # load audio sample
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

## MimiConfig

### class transformers.MimiConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mimi/configuration_mimi.py#L28)

( sampling\_rate = 24000 frame\_rate = None audio\_channels = 1 hidden\_size = 512 num\_filters = 64 num\_residual\_layers = 1 upsampling\_ratios = None kernel\_size = 7 last\_kernel\_size = 3 residual\_kernel\_size = 3 dilation\_growth\_rate = 2 use\_causal\_conv = True pad\_mode = 'constant' compress = 2 trim\_right\_ratio = 1.0 codebook\_size = 2048 codebook\_dim = 256 num\_quantizers = 32 use\_conv\_shortcut = False vector\_quantization\_hidden\_dimension = 256 num\_semantic\_quantizers = 1 upsample\_groups = 512 num\_hidden\_layers = 8 intermediate\_size = 2048 num\_attention\_heads = 8 num\_key\_value\_heads = 8 head\_dim = None hidden\_act = 'gelu' max\_position\_embeddings = 8000 initializer\_range = 0.02 norm\_eps = 1e-05 use\_cache = False use\_streaming = False rope\_theta = 10000.0 sliding\_window = 250 attention\_dropout = 0.0 layer\_scale\_initial\_scale = 0.01 attention\_bias = False \*\*kwargs  )

Parameters

* **sampling\_rate** (`int`, *optional*, defaults to 24000) —
  The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
* **frame\_rate** (`float`, *optional*) —
  Should be computed from the other parameters, yet kept for backward compatibility.
* **audio\_channels** (`int`, *optional*, defaults to 1) —
  Number of channels in the audio data. Either 1 for mono or 2 for stereo.
* **hidden\_size** (`int`, *optional*, defaults to 512) —
  Intermediate representation dimension.
* **num\_filters** (`int`, *optional*, defaults to 64) —
  Number of convolution kernels of first `MimiConv1d` down sampling layer.
* **num\_residual\_layers** (`int`, *optional*, defaults to 1) —
  Number of residual layers.
* **upsampling\_ratios** (`Sequence[int]`, *optional*) —
  Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
  will use the ratios in the reverse order to the ones specified here that must match the decoder order.
  If not specified, will defaults to `[8, 6, 5, 4]`
* **kernel\_size** (`int`, *optional*, defaults to 7) —
  Kernel size for the initial convolution.
* **last\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size for the last convolution layer.
* **residual\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size for the residual layers.
* **dilation\_growth\_rate** (`int`, *optional*, defaults to 2) —
  How much to increase the dilation with each layer.
* **use\_causal\_conv** (`bool`, *optional*, defaults to `True`) —
  Whether to use fully causal convolution.
* **pad\_mode** (`str`, *optional*, defaults to `"constant"`) —
  Padding mode for the convolutions.
* **compress** (`int`, *optional*, defaults to 2) —
  Reduced dimensionality in residual branches.
* **trim\_right\_ratio** (`float`, *optional*, defaults to 1.0) —
  Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
  equal to 1.0, it means that all the trimming is done at the right.
* **codebook\_size** (`int`, *optional*, defaults to 2048) —
  Number of discret codes in each codebooks.
* **codebook\_dim** (`int`, *optional*, defaults to 256) —
  Dimension of the unquantized codebook vectors. If not defined, uses `hidden_size`.
* **num\_quantizers** (`int`, *optional*, defaults to 32) —
  Number of quantizer channels, or codebooks, in the quantizer.
* **use\_conv\_shortcut** (`bool`, *optional*, defaults to `False`) —
  Whether to use a convolutional layer as the ‘skip’ connection in the `MimiResnetBlock` block. If False,
  an identity function will be used, giving a generic residual connection.
* **vector\_quantization\_hidden\_dimension** (`int`, *optional*, defaults to 256) —
  Intermediate representation dimension in the residual vector quantization space.
* **num\_semantic\_quantizers** (`int`, *optional*, defaults to 1) —
  Number of semantic quantizer channels, or codebooks, in the semantic quantizer. Must be lower than `num_quantizers`.
* **upsample\_groups** (`int`, *optional*, defaults to 512) —
  If `frame_rate!=encodec_frame_rate`, indicates the number of groups used in the upsampling operation to go from one rate to another.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 8) —
  Number of hidden layers in the Transformer models.
* **intermediate\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the MLP representations.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
* **head\_dim** (`int`, *optional*, defaults to `hidden_size // num_attention_heads`) —
  The attention head dimension.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 8000) —
  The maximum sequence length that this model might ever be used with. Mimi’s sliding window attention
  allows sequence of up to 8000 tokens.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the LayerNorm normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **use\_streaming** (`bool`, *optional*, defaults to `False`) —
  Whether to use streaming mode. If `True`, the model encode method will return the padding cache that can be used in a subsequent call to the encode method.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **sliding\_window** (`int`, *optional*, defaults to 250) —
  Sliding window attention window size. If not specified, will default to `250`.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **layer\_scale\_initial\_scale** (`float`, *optional*, defaults to 0.01) —
  Initiale scale of the residual rescaling operation done in the Transformer models.
* **attention\_bias** (`bool`, defaults to `False`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.

This is the configuration class to store the configuration of an [MimiModel](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiModel). It is used to instantiate a
Mimi model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
[kyutai/mimi](https://huggingface.co/kyutai/mimi) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MimiModel, MimiConfig

>>> # Initializing a "kyutai/mimi" style configuration
>>> configuration = MimiConfig()

>>> # Initializing a model (with random weights) from the "kyutai/mimi" style configuration
>>> model = MimiModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MimiModel

### class transformers.MimiModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mimi/modeling_mimi.py#L1411)

( config: MimiConfig  )

Parameters

* **config** ([MimiConfig](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mimi neural audio codec model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mimi/modeling_mimi.py#L1631)

( audio\_codes: Tensor padding\_mask: typing.Optional[torch.Tensor] = None decoder\_past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **audio\_codes** (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*) —
  Discret code embeddings computed using `model.encode`.
* **padding\_mask** (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`) —
  Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
  for *masked*.
* **decoder\_past\_key\_values** (`Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio\_codes (those that don’t
  have their past key value states given to this model).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Decodes the given frames into an output audio waveform.

Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
trimmed.

#### encode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mimi/modeling_mimi.py#L1521)

( input\_values: Tensor padding\_mask: typing.Optional[torch.Tensor] = None num\_quantizers: typing.Optional[float] = None encoder\_past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None padding\_cache: typing.Optional[transformers.models.mimi.modeling\_mimi.MimiConv1dPaddingCache] = None use\_streaming: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`) —
  Float values of the input audio waveform.
* **padding\_mask** (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`) —
  Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
  for *masked*.
* **num\_quantizers** (`int`, *optional*) —
  Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
* **encoder\_past\_key\_values** (`Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio\_codes (those that don’t
  have their past key value states given to this model).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Encodes the input audio waveform into discrete codes.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mimi/modeling_mimi.py#L1679)

( input\_values: Tensor padding\_mask: typing.Optional[torch.Tensor] = None num\_quantizers: typing.Optional[int] = None audio\_codes: typing.Optional[torch.Tensor] = None encoder\_past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None decoder\_past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.mimi.modeling_mimi.MimiOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*) —
  Raw audio input converted to Float.
* **padding\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
  for *masked*.
* **num\_quantizers** (`int`, *optional*) —
  Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
* **audio\_codes** (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*) —
  Discret code embeddings computed using `model.encode`.
* **encoder\_past\_key\_values** (`Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio\_codes (those that don’t
  have their past key value states given to this model).
* **decoder\_past\_key\_values** (`Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio\_codes (those that don’t
  have their past key value states given to this model).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.mimi.modeling_mimi.MimiOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.mimi.modeling_mimi.MimiOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MimiConfig](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiConfig)) and inputs.

* **audio\_codes** (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*) — Discret code embeddings computed using `model.encode`.
* **audio\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) — Decoded audio values, obtained using the decoder part of Mimi.
* **encoder\_past\_key\_values** (`Cache`, *optional*) — Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio\_codes (those that don’t
  have their past key value states given to this model).
* **decoder\_past\_key\_values** (`Cache`, *optional*) — Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio\_codes (those that don’t
  have their past key value states given to this model).

The [MimiModel](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from datasets import load_dataset
>>> from transformers import AutoFeatureExtractor, MimiModel

>>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> model_id = "kyutai/mimi"
>>> model = MimiModel.from_pretrained(model_id)
>>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

>>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

>>> outputs = model(**inputs)
>>> audio_codes = outputs.audio_codes
>>> audio_values = outputs.audio_values
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mimi.md)
