# BertJapanese

## Overview

The BERT models trained on Japanese text.

There are models with two different tokenization methods:

- Tokenize with MeCab and WordPiece. This requires some extra dependencies, [fugashi](https://github.com/polm/fugashi) which is a wrapper around [MeCab](https://taku910.github.io/mecab/).
- Tokenize into characters.

To use *MecabTokenizer*, you should `pip install transformers["ja"]` (or `pip install -e .["ja"]` if you install
from source) to install dependencies.

See [details on cl-tohoku repository](https://github.com/cl-tohoku/bert-japanese).

Example of using a model with MeCab and WordPiece tokenization:

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

Example of using a model with Character tokenization:

```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

This model was contributed by [cl-tohoku](https://huggingface.co/cl-tohoku).

This implementation is the same as BERT, except for tokenization method. Refer to [BERT documentation](bert) for
API reference information.

## BertJapaneseTokenizer[[transformers.BertJapaneseTokenizer]]

#### transformers.BertJapaneseTokenizer[[transformers.BertJapaneseTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert_japanese/tokenization_bert_japanese.py#L59)

Construct a BERT tokenizer for Japanese text.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer
to: this superclass for more information regarding those methods.

convert_tokens_to_stringtransformers.BertJapaneseTokenizer.convert_tokens_to_stringhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/bert_japanese/tokenization_bert_japanese.py#L257[{"name": "tokens", "val": ""}]
Converts a sequence of tokens (string) in a single string.

**Parameters:**

vocab_file (`str`) : Path to a one-wordpiece-per-line vocabulary file.

spm_file (`str`, *optional*) : Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm or .model extension) that contains the vocabulary.

do_lower_case (`bool`, *optional*, defaults to `True`) : Whether to lower case the input. Only has an effect when do_basic_tokenize=True.

do_word_tokenize (`bool`, *optional*, defaults to `True`) : Whether to do word tokenization.

do_subword_tokenize (`bool`, *optional*, defaults to `True`) : Whether to do subword tokenization.

word_tokenizer_type (`str`, *optional*, defaults to `"basic"`) : Type of word tokenizer. Choose from ["basic", "mecab", "sudachi", "jumanpp"].

subword_tokenizer_type (`str`, *optional*, defaults to `"wordpiece"`) : Type of subword tokenizer. Choose from ["wordpiece", "character", "sentencepiece",].

mecab_kwargs (`dict`, *optional*) : Dictionary passed to the `MecabTokenizer` constructor.

sudachi_kwargs (`dict`, *optional*) : Dictionary passed to the `SudachiTokenizer` constructor.

jumanpp_kwargs (`dict`, *optional*) : Dictionary passed to the `JumanppTokenizer` constructor.
