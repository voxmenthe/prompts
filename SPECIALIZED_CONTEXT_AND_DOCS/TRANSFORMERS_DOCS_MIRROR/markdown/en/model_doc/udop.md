*This model was released on 2022-12-05 and added to Hugging Face Transformers on 2024-03-04.*

# UDOP

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The UDOP model was proposed in [Unifying Vision, Text, and Layout for Universal Document Processing](https://huggingface.co/papers/2212.02623) by Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal.
UDOP adopts an encoder-decoder Transformer architecture based on [T5](t5) for document AI tasks like document image classification, document parsing and document visual question answering.

The abstract from the paper is the following:

We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 9 Document AI tasks, e.g., document understanding and QA, across diverse data domains like finance reports, academic papers, and websites. UDOP ranks first on the leaderboard of the Document Understanding Benchmark (DUE).\*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/udop_architecture.jpg) UDOP architecture. Taken from the [original paper.](https://huggingface.co/papers/2212.02623)

## Usage tips

* In addition to *input\_ids*, [UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration) also expects the input `bbox`, which are
  the bounding boxes (i.e. 2D-positions) of the input tokens. These can be obtained using an external OCR engine such
  as Google’s [Tesseract](https://github.com/tesseract-ocr/tesseract) (there’s a [Python wrapper](https://pypi.org/project/pytesseract/) available). Each bounding box should be in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the
  position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000
  scale. To normalize, you can use the following function:


```
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

Here, `width` and `height` correspond to the width and height of the original document in which the token
occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:


```
from PIL import Image

# Document can be a png, jpg, etc. PDFs must be converted to images.
image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size
```

One can use [UdopProcessor](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopProcessor) to prepare images and text for the model, which takes care of all of this. By default, this class uses the Tesseract engine to extract a list of words and boxes (coordinates) from a given document. Its functionality is equivalent to that of [LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor), hence it supports passing either `apply_ocr=False` in case you prefer to use your own OCR engine or `apply_ocr=True` in case you want the default OCR engine to be used. Refer to the [usage guide of LayoutLMv2](layoutlmv2#usage-layoutlmv2processor) regarding all possible use cases (the functionality of `UdopProcessor` is identical).

* If using an own OCR engine of choice, one recommendation is Azure’s [Read API](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-read-api), which supports so-called line segments. Use of segment position embeddings typically results in better performance.
* At inference time, it’s recommended to use the `generate` method to autoregressively generate text given a document image.
* The model has been pre-trained on both self-supervised and supervised objectives. One can use the various task prefixes (prompts) used during pre-training to test out the out-of-the-box capabilities. For instance, the model can be prompted with “Question answering. What is the date?”, as “Question answering.” is the task prefix used during pre-training for DocVQA. Refer to the [paper](https://huggingface.co/papers/2212.02623) (table 1) for all task prefixes.
* One can also fine-tune [UdopEncoderModel](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopEncoderModel), which is the encoder-only part of UDOP, which can be seen as a LayoutLMv3-like Transformer encoder. For discriminative tasks, one can just add a linear classifier on top of it and fine-tune it on a labeled dataset.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/microsoft/UDOP).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with UDOP. If
you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll
review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

* Demo notebooks regarding UDOP can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/UDOP) that show how
  to fine-tune UDOP on a custom dataset as well as inference. 🌎
* [Document question answering task guide](../tasks/document_question_answering)

## UdopConfig

### class transformers.UdopConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/configuration_udop.py#L24)

( vocab\_size = 33201 d\_model = 1024 d\_kv = 64 d\_ff = 4096 num\_layers = 24 num\_decoder\_layers = None num\_heads = 16 relative\_attention\_num\_buckets = 32 relative\_attention\_max\_distance = 128 relative\_bias\_args = [{'type': '1d'}, {'type': 'horizontal'}, {'type': 'vertical'}] dropout\_rate = 0.1 layer\_norm\_epsilon = 1e-06 initializer\_factor = 1.0 feed\_forward\_proj = 'relu' is\_encoder\_decoder = True use\_cache = True pad\_token\_id = 0 eos\_token\_id = 1 max\_2d\_position\_embeddings = 1024 image\_size = 224 patch\_size = 16 num\_channels = 3 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 33201) —
  Vocabulary size of the UDOP model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration).
* **d\_model** (`int`, *optional*, defaults to 1024) —
  Size of the encoder layers and the pooler layer.
* **d\_kv** (`int`, *optional*, defaults to 64) —
  Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
  be defined as `num_heads * d_kv`.
* **d\_ff** (`int`, *optional*, defaults to 4096) —
  Size of the intermediate feed forward layer in each `UdopBlock`.
* **num\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder and decoder.
* **num\_decoder\_layers** (`int`, *optional*) —
  Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
* **num\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder and decoder.
* **relative\_attention\_num\_buckets** (`int`, *optional*, defaults to 32) —
  The number of buckets to use for each attention layer.
* **relative\_attention\_max\_distance** (`int`, *optional*, defaults to 128) —
  The maximum distance of the longer sequences for the bucket separation.
* **relative\_bias\_args** (`list[dict]`, *optional*, defaults to `[{'type' -- '1d'}, {'type': 'horizontal'}, {'type': 'vertical'}]`):
  A list of dictionaries containing the arguments for the relative bias layers.
* **dropout\_rate** (`float`, *optional*, defaults to 0.1) —
  The ratio for all dropout layers.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).
* **feed\_forward\_proj** (`string`, *optional*, defaults to `"relu"`) —
  Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. Udopv1.1 uses the
  `"gated-gelu"` feed forward projection. Original Udop uses `"relu"`.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model should behave as an encoder/decoder or not.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the padding token in the vocabulary.
* **eos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the end-of-sequence token in the vocabulary.
* **max\_2d\_position\_embeddings** (`int`, *optional*, defaults to 1024) —
  The maximum absolute position embeddings for relative position encoding.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size of the input images.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The patch size used by the vision encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of channels in the input images.

This is the configuration class to store the configuration of a [UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration). It is used to
instantiate a UDOP model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the UDOP
[microsoft/udop-large](https://huggingface.co/microsoft/udop-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## UdopTokenizer

### class transformers.UdopTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop.py#L152)

( vocab\_file eos\_token = '</s>' unk\_token = '<unk>' sep\_token = '</s>' pad\_token = '<pad>' sep\_token\_box = [1000, 1000, 1000, 1000] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True additional\_special\_tokens = None sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None legacy = True add\_prefix\_space = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **sep\_token\_box** (`list[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) —
  Additional special tokens used by the tokenizer.
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
* **legacy** (`bool`, *optional*, defaults to `True`) —
  Whether or not the `legacy` behaviour of the tokenizer should be used. Legacy is before the merge of #24622
  which includes fixes to properly handle tokens that appear after special tokens. A simple example:
  + `legacy=True`:

Adapted from [LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer) and [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer). Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop.py#L384)

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
adding special tokens. A sequence has the following format:

* single sequence: `X </s>`
* pair of sequences: `A </s> B </s>`

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop.py#L310)

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

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop.py#L361)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
use of token type ids, therefore a list of zeros is returned.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop.py#L492)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## UdopTokenizerFast

### class transformers.UdopTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L148)

( vocab\_file = None tokenizer\_file = None eos\_token = '</s>' sep\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' sep\_token\_box = [1000, 1000, 1000, 1000] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True additional\_special\_tokens = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`, *optional*) —
  Path to the vocabulary file.
* **tokenizer\_file** (`str`, *optional*) —
  Path to the tokenizer file.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **sep\_token\_box** (`list[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) —
  Additional special tokens used by the tokenizer.

Construct a “fast” UDOP tokenizer (backed by HuggingFace’s *tokenizers* library). Adapted from
[LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer) and [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer). Based on
[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### batch\_encode\_plus\_boxes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L438)

( batch\_text\_or\_text\_pairs: typing.Union[list[str], list[tuple[str, str]], list[list[str]]] is\_pair: typing.Optional[bool] = None boxes: typing.Optional[list[list[list[int]]]] = None word\_labels: typing.Optional[list[list[int]]] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  )

Parameters

* **batch\_text\_or\_text\_pairs** (`list[str]`, `list[tuple[str, str]]`, `list[list[str]]`, `list[tuple[list[str], list[str]]]`, and for not-fast tokenizers, also `list[list[int]]`, `list[tuple[list[int], list[int]]]`) —
  Batch of sequences or pair of sequences to be encoded. This can be a list of
  string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
  details in `encode_plus`).

Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

This method is deprecated, `__call__` should be used instead.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L957)

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
adding special tokens. An XLM-RoBERTa sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### call\_boxes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L272)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
* **text\_pair** (`list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
* **boxes** (`list[list[int]]`, `list[list[list[int]]]`) —
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
* **word\_labels** (`list[int]`, `list[list[int]]`, *optional*) —
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
* **return\_special\_tokens\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using
  Python’s tokenizer, this method will raise `NotImplementedError`.
* **return\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the lengths of the encoded inputs.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.
* \***\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **bbox** — List of bounding boxes to be fed to a model.
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **labels** — List of labels to be fed to a model. (when `word_labels` is specified).
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`).

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L982)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.

#### encode\_boxes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L746)

( text: typing.Union[str, list[str], list[int]] text\_pair: typing.Union[str, list[str], list[int], NoneType] = None boxes: typing.Optional[list[list[int]]] = None word\_labels: typing.Optional[list[list[int]]] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  )

Parameters

* **Converts** a string to a sequence of ids (integer), using the tokenizer and vocabulary. Same as doing —
* **`self.convert_tokens_to_ids(self.tokenize(text))`.** —
  text (`str`, `list[str]` or `list[int]`):
  The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
  `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).
  text\_pair (`str`, `list[str]` or `list[int]`, *optional*):
  Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
  the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).

#### encode\_plus\_boxes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/tokenization_udop_fast.py#L789)

( text: typing.Union[str, list[str]] text\_pair: typing.Optional[list[str]] = None boxes: typing.Optional[list[list[int]]] = None word\_labels: typing.Optional[list[list[int]]] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  )

Parameters

* **text** (`str`, `list[str]` or (for non-fast tokenizers) `list[int]`) —
  The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
  `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).
* **text\_pair** (`str`, `list[str]` or `list[int]`, *optional*) —
  Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
  the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).

Tokenize and prepare for the model a sequence or a pair of sequences.

This method is deprecated, `__call__` should be used instead.

## UdopProcessor

### class transformers.UdopProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/processing_udop.py#L55)

( image\_processor tokenizer  )

Parameters

* **image\_processor** (`LayoutLMv3ImageProcessor`) —
  An instance of [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). The image processor is a required input.
* **tokenizer** (`UdopTokenizer` or `UdopTokenizerFast`) —
  An instance of [UdopTokenizer](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopTokenizer) or [UdopTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopTokenizerFast). The tokenizer is a required input.

Constructs a UDOP processor which combines a LayoutLMv3 image processor and a UDOP tokenizer into a single processor.

[UdopProcessor](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopProcessor) offers all the functionalities you need to prepare data for the model.

It first uses [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) to resize, rescale and normalize document images, and optionally applies OCR
to get words and normalized bounding boxes. These are then provided to [UdopTokenizer](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopTokenizer) or [UdopTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopTokenizerFast),
which turns the words and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
classification tasks (such as FUNSD, CORD).

Additionally, it also supports passing `text_target` and `text_pair_target` to the tokenizer, which can be used to
prepare labels for language modeling tasks.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/processing_udop.py#L84)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None text: typing.Union[str, list[str], list[list[str]]] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.udop.processing\_udop.UdopProcessorKwargs]  )

This method first forwards the `images` argument to `~UdopImageProcessor.__call__`. In case
`UdopImageProcessor` was initialized with `apply_ocr` set to `True`, it passes the obtained words and
bounding boxes along with the additional arguments to `__call__()` and returns the output,
together with the prepared `pixel_values`. In case `UdopImageProcessor` was initialized with `apply_ocr` set
to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along with the
additional arguments to `__call__()` and returns the output, together with the prepared
`pixel_values`.

Alternatively, one can pass `text_target` and `text_pair_target` to prepare the targets of UDOP.

Please refer to the docstring of the above two methods for more information.

## UdopModel

### class transformers.UdopModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/modeling_udop.py#L1481)

( config  )

Parameters

* **config** ([UdopModel](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Udop Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/modeling_udop.py#L1524)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None bbox: typing.Optional[dict[str, typing.Any]] = None pixel\_values: typing.Optional[torch.Tensor] = None visual\_bbox: typing.Optional[dict[str, typing.Any]] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None use\_cache = True output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  )

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
* **bbox** (`torch.LongTensor` of shape `({0}, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([UdopProcessor](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopProcessor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).
* **visual\_bbox** (`torch.LongTensor` of shape `(batch_size, patch_sequence_length, 4)`, *optional*) —
  Bounding boxes of each patch in the image. If not provided, bounding boxes are created in the model.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
  [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.
  [What are decoder input IDs?](../glossary#decoder-input-ids) T5 uses the `pad_token_id` as the starting
  token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last
  `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
  `decoder_input_ids` for pretraining take a look at [T5 Training](./t5#training).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_outputs** (`torch.Tensor`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_inputs\_embeds** (`torch.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
  `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **use\_cache** (``, defaults to` True`) -- If set to` True`,` past\_key\_values`key value states are returned and can be used to speed up decoding (see`past\_key\_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

The [UdopModel](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, AutoModel
>>> from datasets import load_dataset
>>> import torch

>>> # load model and processor
>>> # in this case, we already have performed OCR ourselves
>>> # so we initialize the processor with `apply_ocr=False`
>>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
>>> model = AutoModel.from_pretrained("microsoft/udop-large")

>>> # load an example image, along with the words and coordinates
>>> # which were extracted using an OCR engine
>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> words = example["tokens"]
>>> boxes = example["bboxes"]
>>> inputs = processor(image, words, boxes=boxes, return_tensors="pt")

>>> decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

>>> # forward pass
>>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 1, 1024]
```

## UdopForConditionalGeneration

### class transformers.UdopForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/modeling_udop.py#L1673)

( config  )

Parameters

* **config** ([UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The UDOP encoder-decoder Transformer with a language modeling head on top, enabling to generate text given document
images and an optional prompt.

This class is based on [T5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration), extended to deal with images and layout (2D) data.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/modeling_udop.py#L1720)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None bbox: typing.Optional[dict[str, typing.Any]] = None pixel\_values: typing.Optional[torch.Tensor] = None visual\_bbox: typing.Optional[dict[str, typing.Any]] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None use\_cache = True output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None cache\_position: typing.Optional[torch.LongTensor] = None  )

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
* **bbox** (`torch.LongTensor` of shape `({0}, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([UdopProcessor](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopProcessor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).
* **visual\_bbox** (`torch.LongTensor` of shape `(batch_size, patch_sequence_length, 4)`, *optional*) —
  Bounding boxes of each patch in the image. If not provided, bounding boxes are created in the model.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
  [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.
  [What are decoder input IDs?](../glossary#decoder-input-ids) T5 uses the `pad_token_id` as the starting
  token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last
  `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
  `decoder_input_ids` for pretraining take a look at [T5 Training](./t5#training).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_outputs** (`torch.Tensor`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_inputs\_embeds** (`torch.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
  `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **use\_cache** (``, defaults to` True`) -- If set to` True`,` past\_key\_values`key value states are returned and can be used to speed up decoding (see`past\_key\_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

The [UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoProcessor, UdopForConditionalGeneration
>>> from datasets import load_dataset

>>> # load model and processor
>>> # in this case, we already have performed OCR ourselves
>>> # so we initialize the processor with `apply_ocr=False`
>>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
>>> model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large")

>>> # load an example image, along with the words and coordinates
>>> # which were extracted using an OCR engine
>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> words = example["tokens"]
>>> boxes = example["bboxes"]

>>> # one can use the various task prefixes (prompts) used during pre-training
>>> # e.g. the task prefix for DocVQA is "Question answering. "
>>> question = "Question answering. What is the date on the form?"
>>> encoding = processor(image, question, text_pair=words, boxes=boxes, return_tensors="pt")

>>> # autoregressive generation
>>> predicted_ids = model.generate(**encoding)
>>> print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
9/30/92
```

## UdopEncoderModel

### class transformers.UdopEncoderModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/modeling_udop.py#L1884)

( config: UdopConfig  )

Parameters

* **config** ([UdopConfig](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Udop Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/udop/modeling_udop.py#L1926)

( input\_ids: typing.Optional[torch.Tensor] = None bbox: typing.Optional[dict[str, typing.Any]] = None attention\_mask: typing.Optional[torch.Tensor] = None pixel\_values: typing.Optional[torch.Tensor] = None visual\_bbox: typing.Optional[dict[str, typing.Any]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.udop.modeling_udop.BaseModelOutputWithAttentionMask` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
  should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
* **bbox** (`torch.LongTensor` of shape `({0}, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([UdopProcessor](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopProcessor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).
* **visual\_bbox** (`torch.LongTensor` of shape `(batch_size, patch_sequence_length, 4)`, *optional*) —
  Bounding boxes of each patch in the image. If not provided, bounding boxes are created in the model.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
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

`transformers.models.udop.modeling_udop.BaseModelOutputWithAttentionMask` or `tuple(torch.FloatTensor)`

A `transformers.models.udop.modeling_udop.BaseModelOutputWithAttentionMask` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UdopConfig](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model. If `past_key_values` is used only
  the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) — Attention mask used in the model’s forward pass to avoid performing attention on padding token indices.
  Mask values selected in `[0, 1]` — - 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. Contains pre-computed hidden-states (key and values in the
  self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks)
  that can be used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
  the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax,
  used to compute the weighted average in the cross-attention heads.

The [UdopEncoderModel](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, UdopEncoderModel
>>> from huggingface_hub import hf_hub_download
>>> from datasets import load_dataset

>>> # load model and processor
>>> # in this case, we already have performed OCR ourselves
>>> # so we initialize the processor with `apply_ocr=False`
>>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
>>> model = UdopEncoderModel.from_pretrained("microsoft/udop-large")

>>> # load an example image, along with the words and coordinates
>>> # which were extracted using an OCR engine
>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> words = example["tokens"]
>>> boxes = example["bboxes"]
>>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

>>> outputs = model(**encoding)
>>> last_hidden_states = outputs.last_hidden_state
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/udop.md)
