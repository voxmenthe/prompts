*This model was released on 2019-03-24 and added to Hugging Face Transformers on 2020-11-16.*

# BertJapanese

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The BERT models trained on Japanese text.

There are models with two different tokenization methods:

* Tokenize with MeCab and WordPiece. This requires some extra dependencies, [fugashi](https://github.com/polm/fugashi) which is a wrapper around [MeCab](https://taku910.github.io/mecab/).
* Tokenize into characters.

To use *MecabTokenizer*, you should `pip install transformers["ja"]` (or `pip install -e .["ja"]` if you install
from source) to install dependencies.

See [details on cl-tohoku repository](https://github.com/cl-tohoku/bert-japanese).

Example of using a model with MeCab and WordPiece tokenization:


```
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


```
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

## BertJapaneseTokenizer

### class transformers.BertJapaneseTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_japanese/tokenization_bert_japanese.py#L61)

( vocab\_file spm\_file = None do\_lower\_case = False do\_word\_tokenize = True do\_subword\_tokenize = True word\_tokenizer\_type = 'basic' subword\_tokenizer\_type = 'wordpiece' never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' mecab\_kwargs = None sudachi\_kwargs = None jumanpp\_kwargs = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to a one-wordpiece-per-line vocabulary file.
* **spm\_file** (`str`, *optional*) —
  Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm or .model
  extension) that contains the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether to lower case the input. Only has an effect when do\_basic\_tokenize=True.
* **do\_word\_tokenize** (`bool`, *optional*, defaults to `True`) —
  Whether to do word tokenization.
* **do\_subword\_tokenize** (`bool`, *optional*, defaults to `True`) —
  Whether to do subword tokenization.
* **word\_tokenizer\_type** (`str`, *optional*, defaults to `"basic"`) —
  Type of word tokenizer. Choose from [“basic”, “mecab”, “sudachi”, “jumanpp”].
* **subword\_tokenizer\_type** (`str`, *optional*, defaults to `"wordpiece"`) —
  Type of subword tokenizer. Choose from [“wordpiece”, “character”, “sentencepiece”,].
* **mecab\_kwargs** (`dict`, *optional*) —
  Dictionary passed to the `MecabTokenizer` constructor.
* **sudachi\_kwargs** (`dict`, *optional*) —
  Dictionary passed to the `SudachiTokenizer` constructor.
* **jumanpp\_kwargs** (`dict`, *optional*) —
  Dictionary passed to the `JumanppTokenizer` constructor.

Construct a BERT tokenizer for Japanese text.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer
to: this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_japanese/tokenization_bert_japanese.py#L258)

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
adding special tokens. A BERT sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_japanese/tokenization_bert_japanese.py#L250)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_japanese/tokenization_bert_japanese.py#L284)

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bert-japanese.md)
