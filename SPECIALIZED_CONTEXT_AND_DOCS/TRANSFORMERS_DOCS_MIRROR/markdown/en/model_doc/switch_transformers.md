*This model was released on 2021-01-11 and added to Hugging Face Transformers on 2022-11-15.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Switch Transformers

[Switch Transformers](https://huggingface.co/papers/2101.03961) is a sparse T5 model where the MLP layer is replaced by a Mixture-of-Experts (MoE). A routing mechanism associates each token with an expert and each expert is a dense MLP. Sparsity enables better scaling and the routing mechanism allows the model to select relevant weights on the fly which increases model capacity.

You can find all the original Switch Transformers checkpoints under the [Switch Transformer](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f) collection.

This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).

Click on the Switch Transformers models in the right sidebar for more examples of how to apply Switch Transformers to different natural language tasks.

The example below demonstrates how to predict the masked token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation", 
    model="google/switch-base-8",
    dtype=torch.float16,
    device=0
)
print(pipeline("The capital of France is <extra_id_0>."))
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes/) to only quantize the weights to 8-bits.


```
# pip install bitsandbytes
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", device_map="auto", quantization_config=quantization_config)

input_text = "The capital of France is <extra_id_0>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## SwitchTransformersConfig

### class transformers.SwitchTransformersConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/configuration_switch_transformers.py#L24)

( vocab\_size = 32128 d\_model = 768 d\_kv = 64 d\_ff = 2048 expert\_capacity = 64 num\_layers = 12 num\_sparse\_encoder\_layers = 3 num\_decoder\_layers = 12 num\_sparse\_decoder\_layers = 3 num\_heads = 12 num\_experts = 8 router\_bias = False router\_jitter\_noise = 0.01 router\_dtype = 'float32' router\_ignore\_padding\_tokens = False relative\_attention\_num\_buckets = 32 relative\_attention\_max\_distance = 128 dropout\_rate = 0.1 layer\_norm\_epsilon = 1e-06 router\_z\_loss\_coef = 0.001 router\_aux\_loss\_coef = 0.001 initializer\_factor = 1.0 dense\_act\_fn = 'relu' is\_encoder\_decoder = True add\_router\_probs = False use\_cache = True pad\_token\_id = 0 eos\_token\_id = 1 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32128) —
  Vocabulary size of the SwitchTransformers model. Defines the number of different tokens that can be
  represented by the `inputs_ids` passed when calling [SwitchTransformersModel](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel).
* **d\_model** (`int`, *optional*, defaults to 768) —
  Size of the encoder layers and the pooler layer.
* **d\_kv** (`int`, *optional*, defaults to 64) —
  Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model // num_heads`.
* **d\_ff** (`int`, *optional*, defaults to 2048) —
  Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
* **expert\_capacity** (`int`, *optional*, defaults to 64) —
  Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
  Transformer.
* **num\_layers** (`int`, *optional*, defaults to 12) —
  Number of dense hidden layers in the Transformer encoder layer.
* **num\_sparse\_encoder\_layers** (`int`, *optional*, defaults to 3) —
  Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.
* **num\_decoder\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
* **num\_sparse\_decoder\_layers** (`int`, *optional*, defaults to 3) —
  Number of sparse (MoE) dense hidden layers in the Transformer decoder layer.
* **num\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_experts** (`int`, *optional*, defaults to 8) —
  Number of experts for each SwitchTransformer layer.
* **router\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the router.
* **router\_jitter\_noise** (`float`, *optional*, defaults to 0.01) —
  Amount of noise to add to the router.
* **router\_dtype** (`str`, *optional*, default to `"float32"`) —
  The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
  *selective precision* discussion in [the paper](https://huggingface.co/papers/2101.03961).
* **router\_ignore\_padding\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether to ignore padding tokens when routing.
* **relative\_attention\_num\_buckets** (`int`, *optional*, defaults to 32) —
  The number of buckets to use for each attention layer.
* **relative\_attention\_max\_distance** (`int`, *optional*, defaults to 128) —
  The maximum distance of the longer sequences for the bucket separation.
* **dropout\_rate** (`float`, *optional*, defaults to 0.1) —
  The ratio for all dropout layers.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-6) —
  The epsilon used by the layer normalization layers.
* **router\_z\_loss\_coef** (`float`, *optional*, defaults to 0.001) —
  The z loss factor for the total loss.
* **router\_aux\_loss\_coef** (`float`, *optional*, defaults to 0.001) —
  The aux loss factor for the total loss.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).
* **dense\_act\_fn** (`string`, *optional*, defaults to `"relu"`) —
  Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. SwitchTransformersv1.1
  uses the `"gated-gelu"` feed forward projection. Original SwitchTransformers uses `"relu"`.
* **add\_router\_probs** (`bool`, *optional*, defaults to `False`) —
  Whether to output router probabilities to compute router auxiliary loss.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [SwitchTransformersModel](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel). It is used to
instantiate a SwitchTransformers model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the
SwitchTransformers [google/switch-base-8](https://huggingface.co/google/switch-base-8) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SwitchTransformersTop1Router

### class transformers.SwitchTransformersTop1Router

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L126)

( config: SwitchTransformersConfig  )

Router using tokens choose top-1 experts assignment.

This router uses the same mechanism as in Switch Transformer (<https://huggingface.co/papers/2101.03961>) and V-MoE
(<https://huggingface.co/papers/2106.05974>): tokens choose their top experts. Items are sorted by router\_probs and then
routed to their choice of expert until the expert’s expert\_capacity is reached. **There is no guarantee that each
token is processed by an expert**, or that each expert receives at least one token.

#### \_compute\_router\_probabilities

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L146)

( hidden\_states: Tensor  ) → router\_probabilities (`torch.Tensor`)

Parameters

* **hidden\_states** (`torch.Tensor`) —
  (batch\_size, sequence\_length, hidden\_dim) from which router probabilities are computed.

Returns

router\_probabilities (`torch.Tensor`)

Tensor of shape (batch\_size, sequence\_length, num\_experts) corresponding to the probabilities for each
token and expert. Used for routing tokens to experts.
router\_logits (`torch.Tensor`):
Logits tensor of shape (batch\_size, sequence\_length, num\_experts) corresponding to raw router logits.
This is used later for computing router z-loss.

Computes router probabilities from input hidden states.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L187)

( hidden\_states: Tensor  )

Parameters

* **hidden\_states** (`torch.Tensor`) —
  [num\_groups, tokens\_per\_group, hidden\_dim] inputs to send to experts.

Generic forward function for every Router class. Each Router expects to have the same input hidden states
(`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
`router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

## SwitchTransformersSparseMLP

### class transformers.SwitchTransformersSparseMLP

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L268)

( config: SwitchTransformersConfig expert\_class: Module = <class 'transformers.models.switch\_transformers.modeling\_switch\_transformers.SwitchTransformersDenseActDense'>  )

Implementation of the Switch Transformers Sparse MLP module.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L283)

( hidden\_states  )

Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
expert the corresponding hidden states.

## SwitchTransformersModel

### class transformers.SwitchTransformersModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1224)

( config: SwitchTransformersConfig  )

Parameters

* **config** ([SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Switch Transformers Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1272)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_router\_logits: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → `transformers.modeling_outputs.Seq2SeqMoEModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. SWITCH\_TRANSFORMERS is a model with relative position
  embeddings so you should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  [What are input IDs?](../glossary#input-ids)

  To know more on how to prepare `input_ids` for pretraining take a look a [SWITCH\_TRANSFORMERS
  Training](./switch_transformers#training).
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  SWITCH\_TRANSFORMERS uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).

  To know more on how to prepare `decoder_input_ids` for pretraining take a look at [SWITCH\_TRANSFORMERS
  Training](./switch_transformers#training).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
  `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_router\_logits** (`bool`, *optional*) —
  Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
  should not be returned during inference.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.modeling_outputs.Seq2SeqMoEModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.Seq2SeqMoEModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_router\_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **encoder\_router\_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the encoder model, useful to compute the auxiliary loss and the z\_loss for the sparse
  modules.

The [SwitchTransformersModel](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, SwitchTransformersModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
>>> model = SwitchTransformersModel.from_pretrained("google/switch-base-8")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

>>> # preprocess: Prepend decoder_input_ids with start token which is pad token for SwitchTransformersModel.
>>> # This is not needed for torch's SwitchTransformersForConditionalGeneration as it does this internally using labels arg.
>>> decoder_input_ids = model._shift_right(decoder_input_ids)

>>> # forward pass
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
>>> last_hidden_states = outputs.last_hidden_state
```

## SwitchTransformersForConditionalGeneration

### class transformers.SwitchTransformersForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1436)

( config: SwitchTransformersConfig  )

Parameters

* **config** ([SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SWITCH\_TRANSFORMERS Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1484)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.Tensor]]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_router\_logits: typing.Optional[bool] = True return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → `transformers.modeling_outputs.Seq2SeqMoEOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. SWITCH\_TRANSFORMERS is a model with relative position
  embeddings so you should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  [What are input IDs?](../glossary#input-ids)

  To know more on how to prepare `input_ids` for pretraining take a look a [SWITCH\_TRANSFORMERS
  Training](./switch_transformers#training).
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  SWITCH\_TRANSFORMERS uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).

  To know more on how to prepare `decoder_input_ids` for pretraining take a look at [SWITCH\_TRANSFORMERS
  Training](./switch_transformers#training).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
  `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_router\_logits** (`bool`, *optional*, defaults to `True`) —
  Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
  should not be returned during inference.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.modeling_outputs.Seq2SeqMoEOutput` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.Seq2SeqMoEOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_router\_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **encoder\_router\_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the encoder model, useful to compute the auxiliary loss and z\_loss for Mixture of Experts
  models.

The [SwitchTransformersForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
>>> model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8")

>>> # training
>>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
>>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
>>> outputs = model(input_ids=input_ids, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits

>>> # inference
>>> input_ids = tokenizer(
...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> outputs = model.generate(input_ids)
>>> # . To, let’s say you have a dog. To summarize:
>>> # Since the model has been trained on MLM, this will output gibberish
```

## SwitchTransformersEncoderModel

### class transformers.SwitchTransformersEncoderModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1720)

( config: SwitchTransformersConfig  )

Parameters

* **config** ([SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare SWITCH\_TRANSFORMERS Model transformer outputting encoder’s raw hidden-states without any specific head

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1760)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_router\_logits: typing.Optional[bool] = True return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.MoEModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. SWITCH\_TRANSFORMERS is a model with relative position
  embeddings so you should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  To know more on how to prepare `input_ids` for pretraining take a look a [SWITCH\_TRANSFORMERS
  Training](./switch_transformers#training).
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_router\_logits** (`bool`, *optional*, defaults to `True`) —
  Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
  should not be returned during inference.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.MoEModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.MoEModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **router\_probs** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router probabilities that are computed by MoE routers, these terms are used to compute the auxiliary
  loss and the z\_loss for Mixture of Experts models.

The [SwitchTransformersEncoderModel](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, SwitchTransformersEncoderModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
>>> model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8")
>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> outputs = model(input_ids=input_ids)
>>> last_hidden_states = outputs.last_hidden_state
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/switch_transformers.md)
