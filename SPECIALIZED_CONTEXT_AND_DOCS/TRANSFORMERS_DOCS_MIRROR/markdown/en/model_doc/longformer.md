*This model was released on 2020-04-10 and added to Hugging Face Transformers on 2020-11-16.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Longformer

[Longformer](https://huggingface.co/papers/2004.05150) is a transformer model designed for processing long documents. The self-attention operation usually scales quadratically with sequence length, preventing transformers from processing longer sequences. The Longformer attention mechanism overcomes this by scaling linearly with sequence length. It combines local windowed attention with task-specific global attention, enabling efficient processing of documents with thousands of tokens.

You can find all the original Longformer checkpoints under the [Ai2](https://huggingface.co/allenai?search_models=longformer) organization.

Click on the Longformer models in the right sidebar for more examples of how to apply Longformer to different language tasks.

The example below demonstrates how to fill the `<mask>` token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="allenai/longformer-base-4096",
    dtype=torch.float16,
    device=0
)
pipeline("""San Francisco 49ers cornerback Shawntae Spencer will miss the rest of the <mask> with a torn ligament in his left knee.
Spencer, a fifth-year pro, will be placed on injured reserve soon after undergoing surgery Wednesday to repair the ligament. He injured his knee late in the 49ers’ road victory at Seattle on Sept. 14, and missed last week’s victory over Detroit.
Tarell Brown and Donald Strickland will compete to replace Spencer with the 49ers, who kept 12 defensive backs on their 53-man roster to start the season. Brown, a second-year pro, got his first career interception last weekend while filling in for Strickland, who also sat out with a knee injury.""")
```

## Notes

* Longformer is based on [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta) and doesn’t have `token_type_ids`. You don’t need to indicate which token belongs to which segment. You only need to separate the segments with the separation token `</s>` or `tokenizer.sep_token`.
* You can set which tokens can attend locally and which tokens attend globally with the `global_attention_mask` at inference (see this [example](https://huggingface.co/docs/transformers/en/model_doc/longformer#transformers.LongformerModel.forward.example) for more details). A value of `0` means a token attends locally and a value of `1` means a token attends globally.
* [LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM) is trained like [RobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM) and should be used as shown below.


  ```
    input_ids = tokenizer.encode("This is a sentence from [MASK] training data", return_tensors="pt")
    mlm_labels = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
    loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]
  ```

## LongformerConfig

### class transformers.LongformerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/configuration_longformer.py#L34)

( attention\_window: typing.Union[list[int], int] = 512 sep\_token\_id: int = 2 pad\_token\_id: int = 1 bos\_token\_id: int = 0 eos\_token\_id: int = 2 vocab\_size: int = 30522 hidden\_size: int = 768 num\_hidden\_layers: int = 12 num\_attention\_heads: int = 12 intermediate\_size: int = 3072 hidden\_act: str = 'gelu' hidden\_dropout\_prob: float = 0.1 attention\_probs\_dropout\_prob: float = 0.1 max\_position\_embeddings: int = 512 type\_vocab\_size: int = 2 initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-12 onnx\_export: bool = False \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Longformer model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel) or `TFLongformerModel`.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel) or
  `TFLongformerModel`.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **attention\_window** (`int` or `list[int]`, *optional*, defaults to 512) —
  Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
  different window size for each layer, use a `list[int]` where `len(attention_window) == num_hidden_layers`.

This is the configuration class to store the configuration of a [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel) or a `TFLongformerModel`. It
is used to instantiate a Longformer model according to the specified arguments, defining the model architecture.

This is the configuration class to store the configuration of a [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel). It is used to instantiate an
Longformer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the LongFormer
[allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) architecture with a sequence
length 4,096.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LongformerConfig, LongformerModel

>>> # Initializing a Longformer configuration
>>> configuration = LongformerConfig()

>>> # Initializing a model from the configuration
>>> model = LongformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LongformerTokenizer

### class transformers.LongformerTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer.py#L75)

( vocab\_file merges\_file errors = 'replace' bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' add\_prefix\_space = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **errors** (`str`, *optional*, defaults to `"replace"`) —
  Paradigm to follow when decoding bytes to UTF-8. See
  [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word. (Longformer tokenizer detect beginning of words by the preceding space).

Constructs a Longformer tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:


```
>>> from transformers import LongformerTokenizer

>>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer.py#L320)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Longformer sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer.py#L285)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer.py#L372)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not
make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer.py#L345)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`list[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

## LongformerTokenizerFast

### class transformers.LongformerTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer_fast.py#L34)

( vocab\_file = None merges\_file = None tokenizer\_file = None errors = 'replace' bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' add\_prefix\_space = False trim\_offsets = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **errors** (`str`, *optional*, defaults to `"replace"`) —
  Paradigm to follow when decoding bytes to UTF-8. See
  [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word. (Longformer tokenizer detect beginning of words by the preceding space).
* **trim\_offsets** (`bool`, *optional*, defaults to `True`) —
  Whether the post processing step should trim offsets to avoid including whitespaces.

Construct a “fast” Longformer tokenizer (backed by HuggingFace’s *tokenizers* library), derived from the GPT-2
tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:


```
>>> from transformers import LongformerTokenizerFast

>>> tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/tokenization_longformer_fast.py#L241)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not
make use of token type ids, therefore a list of zeros is returned.

## Longformer specific outputs

### class transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L43)

( last\_hidden\_state: FloatTensor hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`) —
  Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for Longformer’s outputs, with potential hidden states, local and global attentions.

### class transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L81)

( last\_hidden\_state: FloatTensor pooler\_output: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`) —
  Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) —
  Last layer hidden-state of the first token of the sequence (classification token) further processed by a
  Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
  prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for Longformer’s outputs that also contains a pooling of the last hidden states.

### class transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L124)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) —
  Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for masked language models outputs.

### class transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L167)

( loss: typing.Optional[torch.FloatTensor] = None start\_logits: typing.Optional[torch.FloatTensor] = None end\_logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) —
  Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) —
  Span-end scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for outputs of question answering Longformer models.

### class transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L209)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) —
  Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for outputs of sentence classification models.

### class transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L252)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) —
  Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) —
  *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for outputs of multiple choice Longformer models.

### class transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L297)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None global\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) —
  Classification scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

Base class for outputs of token classification models.

## LongformerModel

### class transformers.LongformerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1376)

( config add\_pooling\_layer = True  )

Parameters

* **config** ([LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Longformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1492)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None global\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to decide the attention given on each token, local attention or global attention. Tokens with global
  attention attends to all other tokens, and all other tokens attend to them. This is important for
  task-specific finetuning because it makes the model more flexible at representing the task. For example,
  for classification, the  ~~token should be given global attention. For QA, all question tokens should also
  have global attention. Please refer to the [Longformer paper](https://huggingface.co/papers/2004.05150) for more
  details. Mask values selected in `[0, 1]`:~~
  + 0 for local attention (a sliding window attention),
  + 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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

[transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.models.longformer.modeling\_longformer.LongformerBaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a
  Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
  prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

The [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import LongformerModel, AutoTokenizer

>>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

>>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
>>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

>>> attention_mask = torch.ones(
...     input_ids.shape, dtype=torch.long, device=input_ids.device
... )  # initialize to local attention
>>> global_attention_mask = torch.zeros(
...     input_ids.shape, dtype=torch.long, device=input_ids.device
... )  # initialize to global attention to be deactivated for all tokens
>>> global_attention_mask[
...     :,
...     [
...         1,
...         4,
...         21,
...     ],
... ] = 1  # Set global attention to random tokens for the sake of this example
>>> # Usually, set global attention based on the task. For example,
>>> # classification: the <s> token
>>> # QA: question tokens
>>> # LM: potentially on the beginning of sentences and paragraphs
>>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
>>> sequence_output = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output
```

## LongformerForMaskedLM

### class transformers.LongformerForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1625)

( config  )

Parameters

* **config** ([LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Longformer Model with a `language modeling` head on top.”

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1643)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None global\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to decide the attention given on each token, local attention or global attention. Tokens with global
  attention attends to all other tokens, and all other tokens attend to them. This is important for
  task-specific finetuning because it makes the model more flexible at representing the task. For example,
  for classification, the  ~~token should be given global attention. For QA, all question tokens should also
  have global attention. Please refer to the [Longformer paper](https://huggingface.co/papers/2004.05150) for more
  details. Mask values selected in `[0, 1]`:~~
  + 0 for local attention (a sliding window attention),
  + 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.longformer.modeling\_longformer.LongformerMaskedLMOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

The [LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example Mask filling:


```
>>> from transformers import AutoTokenizer, LongformerForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
```

Let’s try a very long input.


```
>>> TXT = (
...     "My friends are <mask> but they eat too many carbs."
...     + " That's why I decide not to eat with them." * 300
... )
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)

>>> tokenizer.decode(predictions).split()
['healthy', 'skinny', 'thin', 'good', 'vegetarian']
```

## LongformerForSequenceClassification

### class transformers.LongformerForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1744)

( config  )

Parameters

* **config** ([LongformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1756)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None global\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to decide the attention given on each token, local attention or global attention. Tokens with global
  attention attends to all other tokens, and all other tokens attend to them. This is important for
  task-specific finetuning because it makes the model more flexible at representing the task. For example,
  for classification, the  ~~token should be given global attention. For QA, all question tokens should also
  have global attention. Please refer to the [Longformer paper](https://huggingface.co/papers/2004.05150) for more
  details. Mask values selected in `[0, 1]`:~~
  + 0 for local attention (a sliding window attention),
  + 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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

[transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.longformer.modeling\_longformer.LongformerSequenceClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

The [LongformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, LongformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, LongformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = LongformerForSequenceClassification.from_pretrained(
...     "allenai/longformer-base-4096", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## LongformerForMultipleChoice

### class transformers.LongformerForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L2080)

( config  )

Parameters

* **config** ([LongformerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMultipleChoice)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L2091)

( input\_ids: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None global\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Mask to decide the attention given on each token, local attention or global attention. Tokens with global
  attention attends to all other tokens, and all other tokens attend to them. This is important for
  task-specific finetuning because it makes the model more flexible at representing the task. For example,
  for classification, the  ~~token should be given global attention. For QA, all question tokens should also
  have global attention. Please refer to the [Longformer paper](https://huggingface.co/papers/2004.05150) for more
  details. Mask values selected in `[0, 1]`:~~
  + 0 for local attention (a sliding window attention),
  + 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
  `input_ids` above)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) —
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

[transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.longformer.modeling\_longformer.LongformerMultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

The [LongformerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, LongformerForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForMultipleChoice.from_pretrained("allenai/longformer-base-4096")

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

## LongformerForTokenClassification

### class transformers.LongformerForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1998)

( config  )

Parameters

* **config** ([LongformerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Longformer transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L2010)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None global\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerTokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to decide the attention given on each token, local attention or global attention. Tokens with global
  attention attends to all other tokens, and all other tokens attend to them. This is important for
  task-specific finetuning because it makes the model more flexible at representing the task. For example,
  for classification, the  ~~token should be given global attention. For QA, all question tokens should also
  have global attention. Please refer to the [Longformer paper](https://huggingface.co/papers/2004.05150) for more
  details. Mask values selected in `[0, 1]`:~~
  + 0 for local attention (a sliding window attention),
  + 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerTokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.longformer.modeling\_longformer.LongformerTokenClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerTokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

The [LongformerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, LongformerForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
>>> model = LongformerForTokenClassification.from_pretrained("allenai/longformer-base-4096")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
...

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

## LongformerForQuestionAnswering

### class transformers.LongformerForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1868)

( config  )

Parameters

* **config** ([LongformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Longformer transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longformer/modeling_longformer.py#L1879)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None global\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **global\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to decide the attention given on each token, local attention or global attention. Tokens with global
  attention attends to all other tokens, and all other tokens attend to them. This is important for
  task-specific finetuning because it makes the model more flexible at representing the task. For example,
  for classification, the  ~~token should be given global attention. For QA, all question tokens should also
  have global attention. Please refer to the [Longformer paper](https://huggingface.co/papers/2004.05150) for more
  details. Mask values selected in `[0, 1]`:~~
  + 0 for local attention (a sliding window attention),
  + 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.longformer.modeling\_longformer.LongformerQuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.

  Local attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token in the sequence to every token with
  global attention (first `x` values) and to every token in the attention window (remaining `attention\_window

  + 1`values). Note that the first`x`values refer to tokens with fixed positions in the text, but the remaining`attention\_window + 1`values refer to tokens with relative positions: the attention weight of a token to itself is located at index`x + attention\_window / 2`and the`attention\_window / 2`preceding (succeeding) values are the attention weights to the`attention\_window / 2`preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first`x`attention weights. If a token has global attention, the attention weights to all other tokens in`attentions`is set to 0, the values should be accessed from`global\_attentions`.
* **global\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

The [LongformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, LongformerForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
>>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
>>> encoding = tokenizer(question, text, return_tensors="pt")
>>> input_ids = encoding["input_ids"]

>>> # default is local attention everywhere
>>> # the forward method will automatically set global attention on question tokens
>>> attention_mask = encoding["attention_mask"]

>>> outputs = model(input_ids, attention_mask=attention_mask)
>>> start_logits = outputs.start_logits
>>> end_logits = outputs.end_logits
>>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

>>> answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
>>> answer = tokenizer.decode(
...     tokenizer.convert_tokens_to_ids(answer_tokens)
... )  # remove space prepending space token
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/longformer.md)
