*This model was released on 2023-12-01 and added to Hugging Face Transformers on 2024-03-05.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Mamba

[Mamba](https://huggingface.co/papers/2312.00752) is a selective structured state space model (SSMs) designed to work around Transformers computational inefficiency when dealing with long sequences. It is a completely attention-free architecture, and comprised of a combination of H3 and gated MLP blocks (Mamba block). Mamba’s “content-based reasoning” allows it to focus on specific parts of an input depending on the current token. Mamba also uses a new hardware-aware parallel algorithm to compensate for the lack of convolutional operations. As a result, Mamba has fast inference and can scale to very long sequences.

You can find all the original Mamba checkpoints under the [State Space Models](https://huggingface.co/state-spaces) organization.

This model was contributed by [Molbap](https://huggingface.co/Molbap) and [AntonV](https://huggingface.co/AntonV).
Click on the Mamba models in the right sidebar for more examples of how to apply Mamba to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="state-spaces/mamba-130m-hf",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to 4-bit integers.


```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig

quantization_config = Int4WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf", dtype=torch.bfloat16, quantization_config=quantization_config, device_map="auto",)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

* The current implementation uses the original CUDA kernels. The FlashAttention equivalent implementation is hosted in the [mamba-ssm](https://github.com/state-spaces/mamba) and [causal\_conv1d](https://github.com/Dao-AILab/causal-conv1d) repositories. Make sure to install them if your hardware supports it!
* Mamba stacks `mixer` layers which are equivalent to `Attention` layers. You can find the main logic of Mamba in the `MambaMixer` class.
* The example below demonstrates how to fine-tune Mamba with [PEFT](https://huggingface.co/docs/peft).


  ```
  from datasets import load_dataset
  from trl import SFTConfig, SFTTrainer
  from peft import LoraConfig

  model_id = "state-spaces/mamba-130m-hf"
  dataset = load_dataset("Abirate/english_quotes", split="train")
  training_args = SFTConfig(dataset_text_field="quote")
  lora_config =  LoraConfig(target_modules=["x_proj", "embeddings", "in_proj", "out_proj"])
  trainer = SFTTrainer(
      model=model_id,
      args=training_args,
      train_dataset=dataset,
      peft_config=lora_config,
  )
  trainer.train()
  ```

## MambaCache

### class transformers.MambaCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L59)

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

Cache for mamba model which does not have attention mechanism and key value states.

Example:


```
>>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

>>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

>>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

>>> # Prepare a cache class and pass it to model's forward
>>> past_key_values = MambaCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values
MambaCache()
```

#### update\_conv\_state

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L131)

( layer\_idx: int new\_conv\_state: Tensor cache\_position: LongTensor  )

#### update\_ssm\_state

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L148)

( layer\_idx: int new\_ssm\_state: Tensor  )

#### reset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L153)

( )

## MambaConfig

### class transformers.MambaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/configuration_mamba.py#L26)

( vocab\_size = 50280 hidden\_size = 768 state\_size = 16 num\_hidden\_layers = 32 layer\_norm\_epsilon = 1e-05 pad\_token\_id = 0 bos\_token\_id = 0 eos\_token\_id = 0 expand = 2 conv\_kernel = 4 use\_bias = False use\_conv\_bias = True hidden\_act = 'silu' initializer\_range = 0.1 residual\_in\_fp32 = True time\_step\_rank = 'auto' time\_step\_scale = 1.0 time\_step\_min = 0.001 time\_step\_max = 0.1 time\_step\_init\_scheme = 'random' time\_step\_floor = 0.0001 rescale\_prenorm\_residual = False use\_cache = True use\_mambapy = False \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 50280) —
  Vocabulary size of the MAMBA model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [MambaModel](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaModel).
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
* **use\_mambapy** (`bool`, *optional*, defaults to `False`) —
  Determines the fallback strategy during training if the CUDA-based official implementation of Mamba is not available. If `True`, the mamba.py implementation is used. If `False`, the naive and slower implementation is used. Consider switching to the naive version if memory is limited.

This is the configuration class to store the configuration of a [MambaModel](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaModel). It is used to instantiate a MAMBA
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the MAMBA
[state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MambaConfig, MambaModel

>>> # Initializing a Mamba configuration
>>> configuration = MambaConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = MambaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MambaModel

### class transformers.MambaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L596)

( config  )

Parameters

* **config** ([MambaModel](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Mamba Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L621)

( input\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.LongTensor] = None cache\_params: typing.Optional[transformers.models.mamba.modeling\_mamba.MambaCache] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None  ) → `transformers.models.mamba.modeling_mamba.MambaOutput` or `tuple(torch.FloatTensor)`

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
* **cache\_params** (`MambaCache`, *optional*) —
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

`transformers.models.mamba.modeling_mamba.MambaOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.mamba.modeling_mamba.MambaOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MambaConfig](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **cache\_params** (`~models.mamba.modeling_mamba.MambaCache`, *optional*, defaults to `None`) — The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.

  Includes both the State space model state matrices after the selective scan, and the Convolutional states
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [MambaModel](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MambaLMHeadModel

### class transformers.MambaForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L707)

( config  )

Parameters

* **config** ([MambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mamba/modeling_mamba.py#L784)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_params: typing.Optional[transformers.models.mamba.modeling\_mamba.MambaCache] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → `transformers.models.mamba.modeling_mamba.MambaCausalLMOutput` or `tuple(torch.FloatTensor)`

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
* **cache\_params** (`MambaCache`, *optional*) —
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

`transformers.models.mamba.modeling_mamba.MambaCausalLMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.mamba.modeling_mamba.MambaCausalLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MambaConfig](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **cache\_params** (`~models.mamba.modeling_mamba.MambaCache`, *optional*, defaults to `None`) — The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.

  Includes both the State space model state matrices after the selective scan, and the Convolutional states
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [MambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mamba.md)
