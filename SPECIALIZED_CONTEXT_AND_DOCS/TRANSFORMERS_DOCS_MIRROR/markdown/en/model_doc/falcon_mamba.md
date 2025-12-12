*This model was released on 2024-10-07 and added to Hugging Face Transformers on 2024-08-12.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# FalconMamba

[FalconMamba](https://huggingface.co/papers/2410.05355) is a 7B large language model, available as pretrained and instruction-tuned variants, based on the [Mamba](./mamba). This model implements a pure Mamba design that focuses on computational efficiency while maintaining strong performance. FalconMamba is significantly faster at inference and requires substantially less memory for long sequence generation. The models are pretrained on a diverse 5.8T token dataset including [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), technical content, code, and mathematical data.

You can find the official FalconMamba checkpoints in the [FalconMamba 7B](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a) collection.

Click on the FalconMamba models in the right sidebar for more examples of how to apply FalconMamba to different language tasks.

The examples below demonstrate how to generate text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-mamba-7b-instruct",
    dtype=torch.bfloat16,
    device=0
)
pipeline(
    "Explain the difference between transformers and SSMs",
    max_length=100,
    do_sample=True,
    temperature=0.7
)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.


```
import torch
from transformers import AutoTokenizer, FalconMambaForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = FalconMambaForCausalLM.from_pretrained(
    "tiiuae/falcon-mamba-7b",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)

inputs = tokenizer("Explain the concept of state space models in simple terms", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## FalconMambaCache

### class transformers.FalconMambaCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L66)

( config: PretrainedConfig max\_batch\_size: int dtype: dtype = torch.float16 device: typing.Union[torch.device, str, NoneType] = None  )

Parameters

* **config** (`PretrainedConfig) —
  The configuration file defining the shape-related attributes required to initialize the static cache.
* **max\_batch\_size** (`int`) —
  The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a smaller batch size is used.
* **dtype** (`torch.dtype`, *optional*, defaults to `torch.float16`) —
  The default `dtype` to use when initializing the layer.
* **device** (`torch.device` or `str`, *optional*) —
  The device on which the cache should be initialized. Should be the same as the layer.

Cache for falcon\_mamba model which does not have attention mechanism and key value states.

Example:


```
>>> from transformers import AutoTokenizer, FalconMambaForCausalLM, FalconMambaCache

>>> model = FalconMambaForCausalLM.from_pretrained("state-spaces/falcon_mamba-130m-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/falcon_mamba-130m-hf")

>>> inputs = tokenizer(text="My name is FalconMamba", return_tensors="pt")

>>> # Prepare a cache class and pass it to model's forward
>>> past_key_values = FalconMambaCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values
FalconMambaCache()
```

#### update\_conv\_state

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L138)

( layer\_idx: int new\_conv\_state: Tensor cache\_position: LongTensor  )

#### update\_ssm\_state

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L155)

( layer\_idx: int new\_ssm\_state: Tensor  )

#### reset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L160)

( )

## FalconMambaConfig

### class transformers.FalconMambaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/configuration_falcon_mamba.py#L27)

( vocab\_size = 50280 hidden\_size = 768 state\_size = 16 num\_hidden\_layers = 32 layer\_norm\_epsilon = 1e-05 pad\_token\_id = 0 bos\_token\_id = 0 eos\_token\_id = 0 expand = 2 conv\_kernel = 4 use\_bias = False use\_conv\_bias = True hidden\_act = 'silu' initializer\_range = 0.1 residual\_in\_fp32 = True time\_step\_rank = 'auto' time\_step\_scale = 1.0 time\_step\_min = 0.001 time\_step\_max = 0.1 time\_step\_init\_scheme = 'random' time\_step\_floor = 0.0001 rescale\_prenorm\_residual = False use\_cache = True use\_falcon\_mambapy = False mixer\_rms\_eps = 1e-06 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 50280) —
  Vocabulary size of the FALCON\_MAMBA model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [FalconMambaModel](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the embeddings and hidden states.
* **state\_size** (`int`, *optional*, defaults to 16) — shape of the state space latents.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the model.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-05) —
  The epsilon to use in the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the beginning of sentence token in the vocabulary.
* **eos\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the end of sentence token in the vocabulary.
* **expand** (`int`, *optional*, defaults to 2) — Expanding factor used to determine the intermediate size.
* **conv\_kernel** (`int`, *optional*, defaults to 4) — Size of the convolution kernel.
* **use\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use bias in [“in\_proj”, “out\_proj”] of the mixer block
* **use\_conv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use bias in the convolution layer of the mixer block.
* **hidden\_act** (`str`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **initializer\_range** (`float`, *optional*, defaults to 0.1) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **residual\_in\_fp32** (`bool`, *optional*, defaults to `True`) —
  Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
* **time\_step\_rank** (`Union[int,str]`, *optional*, defaults to `"auto"`) —
  Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
* **time\_step\_scale** (`float`, *optional*, defaults to 1.0) —
  Scale used used to scale `dt_proj.bias`.
* **time\_step\_min** (`float`, *optional*, defaults to 0.001) —
  Minimum `time_step` used to bound `dt_proj.bias`.
* **time\_step\_max** (`float`, *optional*, defaults to 0.1) —
  Maximum `time_step` used to bound `dt_proj.bias`.
* **time\_step\_init\_scheme** (`float`, *optional*, defaults to `"random"`) —
  Init scheme used for `dt_proj.weight`. Should be one of `["random","uniform"]`
* **time\_step\_floor** (`float`, *optional*, defaults to 0.0001) —
  Minimum clamping value of the `dt_proj.bias` layer initialization.
* **rescale\_prenorm\_residual** (`bool`, *optional*, defaults to `False`) —
  Whether or not to rescale `out_proj` weights when initializing.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the cache should be used.
* **use\_falcon\_mambapy** (`bool`, *optional*, defaults to `False`) —
  This argument corresponds to `use_mambapy` in MambaConfig.
  Determines the fallback strategy during training if the CUDA-based official implementation of Mamba is not available. If `True`, the mamba.py implementation is used. If `False`, the naive and slower implementation is used. Consider switching to the naive version if memory is limited.
* **mixer\_rms\_eps** (`float`, *optional*, defaults to 1e-06) —
  The RMS norm epsilon value that is used in the Mixer RMS norm for B, C and dt states.

This is the configuration class to store the configuration of a [FalconMambaModel](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaModel). It is used to instantiate a FALCON\_MAMBA
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the FALCON\_MAMBA
[tiiuae/falcon-mamba-7b](https://huggingface.co/tiiuae/falcon-mamba-7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import FalconMambaConfig, FalconMambaModel

>>> # Initializing a FalconMamba configuration
>>> configuration = FalconMambaConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = FalconMambaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## FalconMambaModel

### class transformers.FalconMambaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L663)

( config  )

Parameters

* **config** ([FalconMambaModel](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Falcon Mamba Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L683)

( input\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.LongTensor] = None cache\_params: typing.Optional[transformers.models.falcon\_mamba.modeling\_falcon\_mamba.FalconMambaCache] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None  ) → `transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **inputs\_embeds** (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **cache\_params** (`FalconMambaCache`, *optional*) —
  If passed along, the model uses the previous state in all the blocks (which will give the output for the
  `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)

Returns

`transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FalconMambaConfig](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **cache\_params** (`~models.falcon_mamba.modeling_falcon_mamba.FalconMambaCache`, *optional*, defaults to `None`) — The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.

  Includes both the State space model state matrices after the selective scan, and the Convolutional states
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [FalconMambaModel](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## FalconMambaLMHeadModel

### class transformers.FalconMambaForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L769)

( config  )

Parameters

* **config** ([FalconMambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The FALCON\_MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L846)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_params: typing.Optional[transformers.models.falcon\_mamba.modeling\_falcon\_mamba.FalconMambaCache] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → `transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaCausalLMOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **cache\_params** (`FalconMambaCache`, *optional*) —
  If passed along, the model uses the previous state in all the blocks (which will give the output for the
  `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaCausalLMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaCausalLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FalconMambaConfig](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **cache\_params** (`~models.falcon_mamba.modeling_falcon_mamba.FalconMambaCache`, *optional*, defaults to `None`) — The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.

  Includes both the State space model state matrices after the selective scan, and the Convolutional states
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [FalconMambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/falcon_mamba.md)
