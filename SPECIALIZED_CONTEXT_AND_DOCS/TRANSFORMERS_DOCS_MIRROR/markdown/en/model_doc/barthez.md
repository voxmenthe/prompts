*This model was released on 2020-10-23 and added to Hugging Face Transformers on 2020-11-27.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# BARThez

[BARThez](https://huggingface.co/papers/2010.12321) is a [BART](./bart) model designed for French language tasks. Unlike existing French BERT models, BARThez includes a pretrained encoder-decoder, allowing it to generate text as well. This model is also available as a multilingual variant, mBARThez, by continuing pretraining multilingual BART on a French corpus.

You can find all of the original BARThez checkpoints under the [BARThez](https://huggingface.co/collections/dascim/barthez-670920b569a07aa53e3b6887) collection.

This model was contributed by [moussakam](https://huggingface.co/moussakam).
Refer to the [BART](./bart) docs for more usage examples.

The example below demonstrates how to predict the `<mask>` token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="moussaKam/barthez",
    dtype=torch.float16,
    device=0
)
pipeline("Les plantes produisent <mask> grâce à un processus appelé photosynthèse.")
```

## BarthezTokenizer

### class transformers.BarthezTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez.py#L39)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
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
* **sp\_model\_kwargs** (`dict`, *optional*) —
  Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
  SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
  to set:
  + `enable_sampling`: Enable subword regularization.
  + `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

    - `nbest_size = {0,1}`: No sampling is performed.
    - `nbest_size > 1`: samples from the nbest\_size results.
    - `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice)
      using forward-filtering-and-backward-sampling algorithm.
  + `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
    BPE-dropout.
* **sp\_model** (`SentencePieceProcessor`) —
  The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

Adapted from [CamembertTokenizer](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertTokenizer) and [BartTokenizer](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartTokenizer). Construct a BARThez tokenizer. Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez.py#L143)

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
adding special tokens. A BARThez sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez.py#L239)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez.py#L196)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez.py#L169)

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

## BarthezTokenizerFast

### class transformers.BarthezTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez_fast.py#L39)

( vocab\_file = None tokenizer\_file = None bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
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
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) —
  Additional special tokens used by the tokenizer.

Adapted from [CamembertTokenizer](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertTokenizer) and [BartTokenizer](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartTokenizer). Construct a “fast” BARThez tokenizer. Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez_fast.py#L125)

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
adding special tokens. A BARThez sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/barthez/tokenization_barthez_fast.py#L151)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/barthez.md)
