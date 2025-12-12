*This model was released on 2020-05-20 and added to Hugging Face Transformers on 2020-11-16.*

# BERTweet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## BERTweet

[BERTweet](https://huggingface.co/papers/2005.10200) shares the same architecture as [BERT-base](./bert), but it’s pretrained like [RoBERTa](./roberta) on English Tweets. It performs really well on Tweet-related tasks like part-of-speech tagging, named entity recognition, and text classification.

You can find all the original BERTweet checkpoints under the [VinAI Research](https://huggingface.co/vinai?search_models=BERTweet) organization.

Refer to the [BERT](./bert) docs for more examples of how to apply BERTweet to different language tasks.

The example below demonstrates how to predict the `<mask>` token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="vinai/bertweet-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

## Notes

* Use the [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer) or [BertweetTokenizer](/docs/transformers/v4.56.2/en/model_doc/bertweet#transformers.BertweetTokenizer) because it’s preloaded with a custom vocabulary adapted to tweet-specific tokens like hashtags (#), mentions (@), emojis, and common abbreviations. Make sure to also install the [emoji](https://pypi.org/project/emoji/) library.
* Inputs should be padded on the right (`padding="max_length"`) because BERT uses absolute position embeddings.

## BertweetTokenizer

### class transformers.BertweetTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L54)

( vocab\_file merges\_file normalization = False bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **normalization** (`bool`, *optional*, defaults to `False`) —
  Whether or not to apply a normalization preprocess.
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

Constructs a BERTweet tokenizer, using Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### add\_from\_file

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L402)

( f  )

Loads a pre-existing dictionary from a text file and adds its symbols to this instance.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L167)

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
adding special tokens. A BERTweet sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L368)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L221)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. BERTweet does
not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L193)

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

#### normalizeToken

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L341)

( token  )

Normalize tokens in a Tweet

#### normalizeTweet

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bertweet/tokenization_bertweet.py#L307)

( tweet  )

Normalize a raw Tweet

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bertweet.md)
