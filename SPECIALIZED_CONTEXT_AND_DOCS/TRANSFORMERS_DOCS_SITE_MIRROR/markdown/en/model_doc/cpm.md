# CPM

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
here: https://github.com/TsinghuaAI/CPM-Generate

CPM's architecture is the same as GPT-2, except for tokenization method. Refer to [GPT-2 documentation](gpt2) for
API reference information.

## CpmTokenizer[[transformers.CpmTokenizer]]

#### transformers.CpmTokenizer[[transformers.CpmTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm.py#L35)

Runs pre-tokenization with Jieba-RS segmentation tool. It is used in CPM models.

build_inputs_with_special_tokenstransformers.CpmTokenizer.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm.py#L231[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`list[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`list[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`list[int]`List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLNet sequence has the following format:

- single sequence: `X  `
- pair of sequences: `A  B  `

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs to which the special tokens will be added.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
#### convert_tokens_to_string[[transformers.CpmTokenizer.convert_tokens_to_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm.py#L226)

Converts a sequence of tokens (strings for sub-words) in a single string.
#### create_token_type_ids_from_sequences[[transformers.CpmTokenizer.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm.py#L284)

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
#### get_special_tokens_mask[[transformers.CpmTokenizer.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm.py#L256)

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

``list[int]``

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

## CpmTokenizerFast[[transformers.CpmTokenizerFast]]

#### transformers.CpmTokenizerFast[[transformers.CpmTokenizerFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm_fast.py#L30)

Runs pre-tokenization with Jieba-RS segmentation tool. It is used in CPM models.

build_inputs_with_special_tokenstransformers.CpmTokenizerFast.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm_fast.py#L147[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`list[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`list[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`list[int]`List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLNet sequence has the following format:

- single sequence: `X  `
- pair of sequences: `A  B  `

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs to which the special tokens will be added.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
#### create_token_type_ids_from_sequences[[transformers.CpmTokenizerFast.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cpm/tokenization_cpm_fast.py#L172)

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
