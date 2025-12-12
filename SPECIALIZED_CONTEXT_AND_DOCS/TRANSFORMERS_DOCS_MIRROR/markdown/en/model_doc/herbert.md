*This model was released on 2020-05-01 and added to Hugging Face Transformers on 2020-11-16.*

# HerBERT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The HerBERT model was proposed in [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://huggingface.co/papers/2005.00630) by Piotr Rybak, Robert Mroczkowski, Janusz Tracz, and
Ireneusz Gawlik. It is a BERT-based Language Model trained on Polish Corpora using only MLM objective with dynamic
masking of whole words.

The abstract from the paper is the following:

*In recent years, a series of Transformer-based models unlocked major improvements in general natural language
understanding (NLU) tasks. Such a fast pace of research would not be possible without general NLU benchmarks, which
allow for a fair comparison of the proposed methods. However, such benchmarks are available only for a handful of
languages. To alleviate this issue, we introduce a comprehensive multi-task benchmark for the Polish language
understanding, accompanied by an online leaderboard. It consists of a diverse set of tasks, adopted from existing
datasets for named entity recognition, question-answering, textual entailment, and others. We also introduce a new
sentiment analysis task for the e-commerce domain, named Allegro Reviews (AR). To ensure a common evaluation scheme and
promote models that generalize to different NLU tasks, the benchmark includes datasets from varying domains and
applications. Additionally, we release HerBERT, a Transformer-based model trained specifically for the Polish language,
which has the best average performance and obtains the best results for three out of nine tasks. Finally, we provide an
extensive evaluation, including several standard baselines and recently proposed, multilingual Transformer-based
models.*

This model was contributed by [rmroczkowski](https://huggingface.co/rmroczkowski). The original code can be found
[here](https://github.com/allegro/HerBERT).

## Usage example


```
>>> from transformers import HerbertTokenizer, RobertaModel

>>> tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

>>> encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors="pt")
>>> outputs = model(encoded_input)

>>> # HerBERT can also be loaded using AutoTokenizer and AutoModel:
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
```

Herbert implementation is the same as `BERT` except for the tokenization method. Refer to [BERT documentation](bert)
for API reference and examples.

## HerbertTokenizer

### class transformers.HerbertTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert.py#L277)

( vocab\_file merges\_file tokenizer\_file = None cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' sep\_token = '</s>' bos\_token = '<s>' do\_lowercase\_and\_remove\_accent = False additional\_special\_tokens = ['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>'] lang2id = None id2lang = None \*\*kwargs  )

Construct a BPE tokenizer for HerBERT.

Peculiarities:

* uses BERT’s pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
  punctuation character will be treated separately.
* Such pretokenized input is BPE subtokenized

This tokenizer inherits from [XLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMTokenizer) which contains most of the methods. Users should refer to the
superclass for more information regarding methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert.py#L511)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s> B </s>`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert.py#L505)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert.py#L539)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`List[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

## HerbertTokenizerFast

### class transformers.HerbertTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert_fast.py#L28)

( vocab\_file = None merges\_file = None tokenizer\_file = None cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' sep\_token = '</s>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.

Construct a “Fast” BPE tokenizer for HerBERT (backed by HuggingFace’s *tokenizers* library).

Peculiarities:

* uses BERT’s pre-tokenizer: BertPreTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of
  a punctuation character will be treated separately.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the methods. Users should refer to the
superclass for more information regarding methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert_fast.py#L74)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An HerBERT, like BERT sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s> B </s>`

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/herbert/tokenization_herbert_fast.py#L101)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`List[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/herbert.md)
