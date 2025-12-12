*This model was released on 2019-09-11 and added to Hugging Face Transformers on 2020-11-16.*

# CTRL

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

CTRL model was proposed in [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://huggingface.co/papers/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and
Richard Socher. It’s a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus
of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

The abstract from the paper is the following:

*Large-scale language models show promising text generation capabilities, but users cannot easily control particular
aspects of the generated text. We release CTRL, a 1.63 billion-parameter conditional transformer language model,
trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were
derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning while
providing more explicit control over text generation. These codes also allow CTRL to predict which parts of the
training data are most likely given a sequence. This provides a potential method for analyzing large amounts of data
via model-based source attribution.*

This model was contributed by [keskarnitishr](https://huggingface.co/keskarnitishr). The original code can be found
[here](https://github.com/salesforce/ctrl).

## Usage tips

* CTRL makes use of control codes to generate text: it requires generations to be started by certain words, sentences
  or links to generate coherent text. Refer to the [original implementation](https://github.com/salesforce/ctrl) for
  more information.
* CTRL is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than
  the left.
* CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text as it can be
  observed in the *run\_generation.py* example script.
* The PyTorch models can take the `past_key_values` as input, which is the previously computed key/value attention pairs.
  Using the `past_key_values` value prevents the model from re-computing
  pre-computed values in the context of text generation. See the [`forward`](model_doc/ctrl#transformers.CTRLModel.forward)
  method for more information on the usage of this argument.

## Resources

* [Text classification task guide](../tasks/sequence_classification)
* [Causal language modeling task guide](../tasks/language_modeling)

## CTRLConfig

### class transformers.CTRLConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/configuration_ctrl.py#L24)

( vocab\_size = 246534 n\_positions = 256 n\_embd = 1280 dff = 8192 n\_layer = 48 n\_head = 16 resid\_pdrop = 0.1 embd\_pdrop = 0.1 layer\_norm\_epsilon = 1e-06 initializer\_range = 0.02 use\_cache = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 246534) —
  Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [CTRLModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel) or `TFCTRLModel`.
* **n\_positions** (`int`, *optional*, defaults to 256) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **n\_embd** (`int`, *optional*, defaults to 1280) —
  Dimensionality of the embeddings and hidden states.
* **dff** (`int`, *optional*, defaults to 8192) —
  Dimensionality of the inner dimension of the feed forward networks (FFN).
* **n\_layer** (`int`, *optional*, defaults to 48) —
  Number of hidden layers in the Transformer encoder.
* **n\_head** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **resid\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **embd\_pdrop** (`int`, *optional*, defaults to 0.1) —
  The dropout ratio for the embeddings.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-06) —
  The epsilon to use in the layer normalization layers
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [CTRLModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel) or a `TFCTRLModel`. It is used to
instantiate a CTRL model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
[Salesforce/ctrl](https://huggingface.co/Salesforce/ctrl) architecture from SalesForce.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import CTRLConfig, CTRLModel

>>> # Initializing a CTRL configuration
>>> configuration = CTRLConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = CTRLModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## CTRLTokenizer

### class transformers.CTRLTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/tokenization_ctrl.py#L110)

( vocab\_file merges\_file unk\_token = '<unk>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.

Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/tokenization_ctrl.py#L215)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## CTRLModel

### class transformers.CTRLModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L234)

( config  )

Parameters

* **config** ([CTRLModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Ctrl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L270)

( input\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) —
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`
  (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

  If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) and
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
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
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig)) and inputs.

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

The [CTRLModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, CTRLModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLModel.from_pretrained("Salesforce/ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 5, 1280]
```

## CTRLLMHeadModel

### class transformers.CTRLLMHeadModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L444)

( config  )

Parameters

* **config** ([CTRLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L455)

( input\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) —
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`
  (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

  If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) and
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
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
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig)) and inputs.

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

The [CTRLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoTokenizer, CTRLLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Wikipedia The llama is", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> sequence_ids = model.generate(inputs["input_ids"])
>>> sequences = tokenizer.batch_decode(sequence_ids)
>>> sequences
['Wikipedia The llama is a member of the family Bovidae. It is native to the Andes of Peru,']

>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> round(outputs.loss.item(), 2)
9.21

>>> list(outputs.logits.shape)
[1, 5, 246534]
```

## CTRLForSequenceClassification

### class transformers.CTRLForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L587)

( config  )

Parameters

* **config** ([CTRLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a sequence classification head on top (linear layer).
[CTRLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last
token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in
each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last
value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L597)

( input\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) —
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`
  (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

  If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) and
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [CTRLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, CTRLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```


```
>>> import torch

>>> torch.manual_seed(42)
>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.93
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, CTRLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLForSequenceClassification.from_pretrained(
...     "Salesforce/ctrl", problem_type="multi_label_classification"
... )

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```


```
>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> num_labels = len(model.config.id2label)
>>> labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
...     torch.float
... )
>>> loss = model(**inputs, labels=labels).loss
>>> loss.backward()
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/ctrl.md)
