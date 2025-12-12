# HerBERT

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

```python
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

## HerbertTokenizer[[transformers.HerbertTokenizer]]

#### transformers.HerbertTokenizer[[transformers.HerbertTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/herbert/tokenization_herbert.py#L30)

Construct a BPE tokenizer for HerBERT (backed by HuggingFace's tokenizers library).

Peculiarities:

- uses BERT's pre-tokenizer: BertPreTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of
  a punctuation character will be treated separately.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the methods. Users should refer to the
superclass for more information regarding methods.

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

merges_file (`str`) : Path to the merges file.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token.

pad_token (`str`, *optional*, defaults to `""`) : The padding token.

mask_token (`str`, *optional*, defaults to `""`) : The mask token.

sep_token (`str`, *optional*, defaults to `""`) : The separator token.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary.

merges (`str` or `list[str]`, *optional*) : Custom merges list.

## HerbertTokenizerFast[[transformers.HerbertTokenizer]]

#### transformers.HerbertTokenizer[[transformers.HerbertTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/herbert/tokenization_herbert.py#L30)

Construct a BPE tokenizer for HerBERT (backed by HuggingFace's tokenizers library).

Peculiarities:

- uses BERT's pre-tokenizer: BertPreTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of
  a punctuation character will be treated separately.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the methods. Users should refer to the
superclass for more information regarding methods.

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

merges_file (`str`) : Path to the merges file.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token.

pad_token (`str`, *optional*, defaults to `""`) : The padding token.

mask_token (`str`, *optional*, defaults to `""`) : The mask token.

sep_token (`str`, *optional*, defaults to `""`) : The separator token.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary.

merges (`str` or `list[str]`, *optional*) : Custom merges list.
