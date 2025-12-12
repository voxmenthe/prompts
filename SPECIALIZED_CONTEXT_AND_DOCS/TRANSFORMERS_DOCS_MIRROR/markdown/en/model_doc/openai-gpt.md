*This model was released on 2018-06-11 and added to Hugging Face Transformers on 2023-06-20.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

# GPT

[GPT (Generative Pre-trained Transformer)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) ([blog post](https://openai.com/index/language-unsupervised/)) focuses on effectively learning text representations and transferring them to tasks. This model trains the Transformer decoder to predict the next word, and then fine-tuned on labeled data.

GPT can generate high-quality text, making it well-suited for a variety of natural language understanding tasks such as textual entailment, question answering, semantic similarity, and document classification.

You can find all the original GPT checkpoints under the [OpenAI community](https://huggingface.co/openai-community/openai-gpt) organization.

Click on the GPT models in the right sidebar for more examples of how to apply GPT to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

generator = pipeline(task="text-generation", model="openai-community/gpt", dtype=torch.float16, device=0)
output = generator("The future of AI is", max_length=50, do_sample=True)
print(output[0]["generated_text"])
```

## Notes

* Inputs should be padded on the right because GPT uses absolute position embeddings.

## OpenAIGPTConfig

### class transformers.OpenAIGPTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/configuration_openai.py#L25)

( vocab\_size = 40478 n\_positions = 512 n\_embd = 768 n\_layer = 12 n\_head = 12 afn = 'gelu' resid\_pdrop = 0.1 embd\_pdrop = 0.1 attn\_pdrop = 0.1 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 summary\_type = 'cls\_index' summary\_use\_proj = True summary\_activation = None summary\_proj\_to\_labels = True summary\_first\_dropout = 0.1 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 40478) —
  Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [OpenAIGPTModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) or `TFOpenAIGPTModel`.
* **n\_positions** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **n\_embd** (`int`, *optional*, defaults to 768) —
  Dimensionality of the embeddings and hidden states.
* **n\_layer** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **n\_head** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **afn** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **resid\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **embd\_pdrop** (`int`, *optional*, defaults to 0.1) —
  The dropout ratio for the embeddings.
* **attn\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-05) —
  The epsilon to use in the layer normalization layers
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **summary\_type** (`str`, *optional*, defaults to `"cls_index"`) —
  Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and
  [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).

  Has to be one of the following options:

  + `"last"`: Take the last token hidden state (like XLNet).
  + `"first"`: Take the first token hidden state (like BERT).
  + `"mean"`: Take the mean of all tokens hidden states.
  + `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
  + `"attn"`: Not implemented now, use multi-head attention.
* **summary\_use\_proj** (`bool`, *optional*, defaults to `True`) —
  Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and
  [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).

  Whether or not to add a projection after the vector extraction.
* **summary\_activation** (`str`, *optional*) —
  Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and
  [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).

  Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
* **summary\_proj\_to\_labels** (`bool`, *optional*, defaults to `True`) —
  Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and
  [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).

  Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
* **summary\_first\_dropout** (`float`, *optional*, defaults to 0.1) —
  Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and
  [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).

  The dropout ratio to be used after the projection and activation.

This is the configuration class to store the configuration of a [OpenAIGPTModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) or a `TFOpenAIGPTModel`. It is
used to instantiate a GPT model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the GPT
[openai-community/openai-gpt](https://huggingface.co/openai-community/openai-gpt) architecture from OpenAI.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import OpenAIGPTConfig, OpenAIGPTModel

>>> # Initializing a GPT configuration
>>> configuration = OpenAIGPTConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = OpenAIGPTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## OpenAIGPTModel

### class transformers.OpenAIGPTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L409)

( config  )

Parameters

* **config** ([OpenAIGPTModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Openai Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L435)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [OpenAIGPTModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

* forward

## OpenAIGPTLMHeadModel

### class transformers.OpenAIGPTLMHeadModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L534)

( config  )

Parameters

* **config** ([OpenAIGPTLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L545)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [OpenAIGPTLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoTokenizer, OpenAIGPTLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
>>> model = OpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

* forward

## OpenAIGPTDoubleHeadsModel

### class transformers.OpenAIGPTDoubleHeadsModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L616)

( config  )

Parameters

* **config** ([OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L630)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None mc\_token\_ids: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None mc\_labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
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
* **mc\_token\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input) —
  Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) - 1]`.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-1, 0, ..., config.vocab_size]` All labels set to `-100` are
  ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
* **mc\_labels** (`torch.LongTensor` of shape `(batch_size)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
  where *num\_choices* is the size of the second dimension of the input tensors. (see *input\_ids* above)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **mc\_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided) — Multiple choice classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **mc\_logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, OpenAIGPTDoubleHeadsModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
>>> model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-community/openai-gpt")
>>> tokenizer.add_special_tokens(
...     {"cls_token": "[CLS]"}
... )  # Add a [CLS] to the vocabulary (we should train it also!)
>>> model.resize_token_embeddings(len(tokenizer))

>>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
>>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
>>> mc_token_ids = torch.tensor([input_ids.size(-1) - 1, input_ids.size(-1) - 1]).unsqueeze(0)  # Batch size 1

>>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
>>> lm_logits = outputs.logits
>>> mc_logits = outputs.mc_logits
```

* forward

## OpenAIGPTForSequenceClassification

### class transformers.OpenAIGPTForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L734)

( config  )

Parameters

* **config** ([OpenAIGPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).
[OpenAIGPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification) uses the last token in order to do the classification, as other causal
models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the
last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding
token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/modeling_openai.py#L744)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
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
elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [OpenAIGPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, OpenAIGPTForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
>>> model = OpenAIGPTForSequenceClassification.from_pretrained("openai-community/openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = OpenAIGPTForSequenceClassification.from_pretrained("openai-community/openai-gpt", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, OpenAIGPTForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
>>> model = OpenAIGPTForSequenceClassification.from_pretrained("openai-community/openai-gpt", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = OpenAIGPTForSequenceClassification.from_pretrained(
...     "openai-community/openai-gpt", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

* forward

## OpenAIGPTTokenizer

### class transformers.OpenAIGPTTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/tokenization_openai.py#L235)

( vocab\_file merges\_file unk\_token = '<unk>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.

Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:

* lowercases all inputs,
* uses `SpaCy` tokenizer and `ftfy` for pre-BPE tokenization if they are installed, fallback to BERT’s
  `BasicTokenizer` if not.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/tokenization_openai.py#L361)

( tokens  )

Converts a sequence of tokens (string) in a single string.

## OpenAIGPTTokenizerFast

### class transformers.OpenAIGPTTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/openai/tokenization_openai_fast.py#L29)

( vocab\_file = None merges\_file = None tokenizer\_file = None unk\_token = '<unk>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.

Construct a “fast” GPT Tokenizer (backed by HuggingFace’s *tokenizers* library). Based on Byte-Pair-Encoding with
the following peculiarities:

* lower case all inputs
* uses BERT’s BasicTokenizer for pre-BPE tokenization

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/openai-gpt.md)
