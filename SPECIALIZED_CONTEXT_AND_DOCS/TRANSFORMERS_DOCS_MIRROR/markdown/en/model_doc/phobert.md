*This model was released on 2020-03-02 and added to Hugging Face Transformers on 2020-11-16.*

# PhoBERT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The PhoBERT model was proposed in [PhoBERT: Pre-trained language models for Vietnamese](https://huggingface.co/papers/2003.00744) by Dat Quoc Nguyen, Anh Tuan Nguyen.

The abstract from the paper is the following:

*We present PhoBERT with two versions, PhoBERT-base and PhoBERT-large, the first public large-scale monolingual
language models pre-trained for Vietnamese. Experimental results show that PhoBERT consistently outperforms the recent
best pre-trained multilingual model XLM-R (Conneau et al., 2020) and improves the state-of-the-art in multiple
Vietnamese-specific NLP tasks including Part-of-speech tagging, Dependency parsing, Named-entity recognition and
Natural language inference.*

This model was contributed by [dqnguyen](https://huggingface.co/dqnguyen). The original code can be found [here](https://github.com/VinAIResearch/PhoBERT).

## Usage example


```
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> phobert = AutoModel.from_pretrained("vinai/phobert-base")
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

>>> # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
>>> line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = phobert(input_ids)  # Models outputs are now tuples
```

PhoBERT implementation is the same as BERT, except for tokenization. Refer to [BERT documentation](bert) for information on
configuration classes and their parameters. PhoBERT-specific tokenizer is documented below.

## PhobertTokenizer

### class transformers.PhobertTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phobert/tokenization_phobert.py#L51)

( vocab\_file merges\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **bos\_token** (`st`, *optional*, defaults to `"<s>"`) —
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

Construct a PhoBERT tokenizer. Based on Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### add\_from\_file

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phobert/tokenization_phobert.py#L327)

( f  )

Loads a pre-existing dictionary from a text file and adds its symbols to this instance.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phobert/tokenization_phobert.py#L146)

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
adding special tokens. A PhoBERT sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phobert/tokenization_phobert.py#L293)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phobert/tokenization_phobert.py#L200)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. PhoBERT does not
make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phobert/tokenization_phobert.py#L172)

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/phobert.md)
