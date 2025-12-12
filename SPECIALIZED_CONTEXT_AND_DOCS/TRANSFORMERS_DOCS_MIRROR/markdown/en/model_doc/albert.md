*This model was released on 2019-09-26 and added to Hugging Face Transformers on 2020-11-16.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# ALBERT

[ALBERT](https://huggingface.co/papers/1909.11942) is designed to address memory limitations of scaling and training of [BERT](./bert). It adds two parameter reduction techniques. The first, factorized embedding parametrization, splits the larger vocabulary embedding matrix into two smaller matrices so you can grow the hidden size without adding a lot more parameters. The second, cross-layer parameter sharing, allows layer to share parameters which keeps the number of learnable parameters lower.

ALBERT was created to address problems like — GPU/TPU memory limitations, longer training times, and unexpected model degradation in BERT. ALBERT uses two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

* **Factorized embedding parameterization:** The large vocabulary embedding matrix is decomposed into two smaller matrices, reducing memory consumption.
* **Cross-layer parameter sharing:** Instead of learning separate parameters for each transformer layer, ALBERT shares parameters across layers, further reducing the number of learnable weights.

ALBERT uses absolute position embeddings (like BERT) so padding is applied at right. Size of embeddings is 128 While BERT uses 768. ALBERT can processes maximum 512 token at a time.

You can find all the original ALBERT checkpoints under the [ALBERT community](https://huggingface.co/albert) organization.

Click on the ALBERT models in the right sidebar for more examples of how to apply ALBERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="albert-base-v2",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.", top_k=5)
```

## Notes

* Inputs should be padded on the right because BERT uses absolute position embeddings.
* The embedding size `E` is different from the hidden size `H` because the embeddings are context independent (one embedding vector represents one token) and the hidden states are context dependent (one hidden state represents a sequence of tokens). The embedding matrix is also larger because `V x E` where `V` is the vocabulary size. As a result, it’s more logical if `H >> E`. If `E < H`, the model has less parameters.

## Resources

The resources provided in the following sections consist of a list of official Hugging Face and community (indicated by 🌎) resources to help you get started with AlBERT. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

Text Classification

* `AlbertForSequenceClassification` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
* Check the [Text classification task guide](../tasks/sequence_classification) on how to use the model.

Token Classification

* `AlbertForTokenClassification` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification).
* [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
* Check the [Token classification task guide](../tasks/token_classification) on how to use the model.

Fill-Mask

* `AlbertForMaskedLM` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
* [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
* Check the [Masked language modeling task guide](../tasks/masked_language_modeling) on how to use the model.

Question Answering

* `AlbertForQuestionAnswering` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
* [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
* Check the [Question answering task guide](../tasks/question_answering) on how to use the model.

**Multiple choice**

* [AlbertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertForMultipleChoice) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
* Check the [Multiple choice task guide](../tasks/multiple_choice) on how to use the model.

## AlbertConfig

### class transformers.AlbertConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/albert/configuration_albert.py#L25)

( vocab\_size = 30000 embedding\_size = 128 hidden\_size = 4096 num\_hidden\_layers = 12 num\_hidden\_groups = 1 num\_attention\_heads = 64 intermediate\_size = 16384 inner\_group\_num = 1 hidden\_act = 'gelu\_new' hidden\_dropout\_prob = 0 attention\_probs\_dropout\_prob = 0 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 classifier\_dropout\_prob = 0.1 position\_embedding\_type = 'absolute' pad\_token\_id = 0 bos\_token\_id = 2 eos\_token\_id = 3 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30000) —
  Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling `AlbertModel` or `TFAlbertModel`.
* **embedding\_size** (`int`, *optional*, defaults to 128) —
  Dimensionality of vocabulary embeddings.
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_hidden\_groups** (`int`, *optional*, defaults to 1) —
  Number of groups for the hidden layers, parameters in the same group are shared.
* **num\_attention\_heads** (`int`, *optional*, defaults to 64) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 16384) —
  The dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **inner\_group\_num** (`int`, *optional*, defaults to 1) —
  The number of inner repetition of attention and ffn.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu_new"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling `AlbertModel` or `TFAlbertModel`.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **classifier\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for attached classifiers.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 2) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 3) —
  End of stream token id.

This is the configuration class to store the configuration of a `AlbertModel` or a `TFAlbertModel`. It is used
to instantiate an ALBERT model according to the specified arguments, defining the model architecture. Instantiating
a configuration with the defaults will yield a similar configuration to that of the ALBERT
[albert/albert-xxlarge-v2](https://huggingface.co/albert/albert-xxlarge-v2) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import AlbertConfig, AlbertModel

>>> # Initializing an ALBERT-xxlarge style configuration
>>> albert_xxlarge_configuration = AlbertConfig()

>>> # Initializing an ALBERT-base style configuration
>>> albert_base_configuration = AlbertConfig(
...     hidden_size=768,
...     num_attention_heads=12,
...     intermediate_size=3072,
... )

>>> # Initializing a model (with random weights) from the ALBERT-base style configuration
>>> model = AlbertModel(albert_xxlarge_configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## AlbertTokenizer

[[autodoc]] AlbertTokenizer - build\_inputs\_with\_special\_tokens - get\_special\_tokens\_mask - create\_token\_type\_ids\_from\_sequences - save\_vocabulary

## AlbertTokenizerFast

### class transformers.AlbertTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/albert/tokenization_albert_fast.py#L38)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True remove\_space = True keep\_accents = False bos\_token = '[CLS]' eos\_token = '[SEP]' unk\_token = '<unk>' sep\_token = '[SEP]' pad\_token = '<pad>' cls\_token = '[CLS]' mask\_token = '[MASK]' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **remove\_space** (`bool`, *optional*, defaults to `True`) —
  Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
* **keep\_accents** (`bool`, *optional*, defaults to `False`) —
  Whether or not to keep accents when tokenizing.
* **bos\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
  that is used for the end of sequence. The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.

Construct a “fast” ALBERT tokenizer (backed by HuggingFace’s *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/albert/tokenization_albert_fast.py#L133)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An ALBERT sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

## Albert specific outputs

### class transformers.models.albert.modeling\_albert.AlbertForPreTrainingOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/albert/modeling_albert.py#L578)

( loss: typing.Optional[torch.FloatTensor] = None prediction\_logits: typing.Optional[torch.FloatTensor] = None sop\_logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) —
  Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
* **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) —
  Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **sop\_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) —
  Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
  before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of `AlbertForPreTraining`.

## AlbertModel

[[autodoc]] AlbertModel - forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining - forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM - forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification - forward

## AlbertForMultipleChoice

### class transformers.AlbertForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/albert/modeling_albert.py#L1237)

( config: AlbertConfig  )

Parameters

* **config** ([AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/albert/modeling_albert.py#L1248)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.albert.modeling\_albert.AlbertForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) and
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where *num\_choices* is the size of the second dimension of the input tensors. (see
  *input\_ids* above)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.albert.modeling\_albert.AlbertForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.albert.modeling\_albert.AlbertForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig)) and inputs.

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
* **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **sop\_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
  before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [AlbertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, AlbertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-xxlarge-v2")
>>> model = AlbertForMultipleChoice.from_pretrained("albert/albert-xxlarge-v2")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

>>> # the linear classifier still needs to be trained
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification - forward

## AlbertForQuestionAnswering

[[autodoc]] AlbertForQuestionAnswering - forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/albert.md)
