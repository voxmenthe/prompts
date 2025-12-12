*This model was released on 2020-12-01 and added to Hugging Face Transformers on 2021-04-10.*

# CPM

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The CPM model was proposed in [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://huggingface.co/papers/2012.00413) by Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin,
Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen,
Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.

The abstract from the paper is the following:

*Pre-trained Language Models (PLMs) have proven to be beneficial for various downstream NLP tasks. Recently, GPT-3,
with 175 billion parameters and 570GB training data, drew a lot of attention due to the capacity of few-shot (even
zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks is still challenging, as the training corpus
of GPT-3 is primarily English, and the parameters are not publicly available. In this technical report, we release the
Chinese Pre-trained Language Model (CPM) with generative pre-training on large-scale Chinese training data. To the best
of our knowledge, CPM, with 2.6 billion parameters and 100GB Chinese training data, is the largest Chinese pre-trained
language model, which could facilitate several downstream Chinese NLP tasks, such as conversation, essay generation,
cloze test, and language understanding. Extensive experiments demonstrate that CPM achieves strong performance on many
NLP tasks in the settings of few-shot (even zero-shot) learning.*

This model was contributed by [canwenxu](https://huggingface.co/canwenxu). The original implementation can be found
here: <https://github.com/TsinghuaAI/CPM-Generate>

CPM’s architecture is the same as GPT-2, except for tokenization method. Refer to [GPT-2 documentation](gpt2) for
API reference information.

## CpmTokenizer

### class transformers.CpmTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm.py#L35)

( vocab\_file do\_lower\_case = False remove\_space = True keep\_accents = False bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' sep\_token = '<sep>' pad\_token = '<pad>' cls\_token = '<cls>' mask\_token = '<mask>' additional\_special\_tokens = ['<eop>', '<eod>'] sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm.py#L241)

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
adding special tokens. An XLNet sequence has the following format:

* single sequence: `X <sep> <cls>`
* pair of sequences: `A <sep> B <sep> <cls>`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm.py#L235)

( tokens  )

Converts a sequence of tokens (strings for sub-words) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm.py#L296)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:


```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm.py#L267)

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

## CpmTokenizerFast

### class transformers.CpmTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm_fast.py#L30)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = False remove\_space = True keep\_accents = False bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' sep\_token = '<sep>' pad\_token = '<pad>' cls\_token = '<cls>' mask\_token = '<mask>' additional\_special\_tokens = ['<eop>', '<eod>'] \*\*kwargs  )

Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm_fast.py#L148)

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
adding special tokens. An XLNet sequence has the following format:

* single sequence: `X <sep> <cls>`
* pair of sequences: `A <sep> B <sep> <cls>`

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cpm/tokenization_cpm_fast.py#L174)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:


```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/cpm.md)
