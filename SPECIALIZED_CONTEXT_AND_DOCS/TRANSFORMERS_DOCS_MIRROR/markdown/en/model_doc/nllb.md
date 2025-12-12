![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

*This model was released on 2022-07-11 and added to Hugging Face Transformers on 2022-07-18.*

# NLLB

[NLLB: No Language Left Behind](https://huggingface.co/papers/2207.04672) is a multilingual translation model. It’s trained on data using data mining techniques tailored for low-resource languages and supports over 200 languages. NLLB features a conditional compute architecture using a Sparsely Gated Mixture of Experts.

You can find all the original NLLB checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=nllb) organization.

This model was contributed by [Lysandre](https://huggingface.co/lysandre).  
Click on the NLLB models in the right sidebar for more examples of how to apply NLLB to different translation tasks.

The example below demonstrates how to translate text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn", dtype=torch.float16, device=0)
pipeline("UN Chief says there is no military solution in Syria")
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 8-bits.


```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")

article = "UN Chief says there is no military solution in Syria"
inputs = tokenizer(article, return_tensors="pt").to(model.device)
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30,
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.


```
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("facebook/nllb-200-distilled-600M")
visualizer("UN Chief says there is no military solution in Syria")
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/NLLB-Attn-Mask.png)

## Notes

* The tokenizer was updated in April 2023 to prefix the source sequence with the source language rather than the target language. This prioritizes zero-shot performance at a minor cost to supervised performance.


  ```
  >>> from transformers import NllbTokenizer

  >>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
  >>> tokenizer("How was your day?").input_ids
  [256047, 13374, 1398, 4260, 4039, 248130, 2]
  ```

  To revert to the legacy behavior, use the code example below.


  ```
  >>> from transformers import NllbTokenizer

  >>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
  ```
* For non-English languages, specify the language’s [BCP-47](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code with the `src_lang` keyword as shown below.
* See example below for a translation from Romanian to German.


  ```
  >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

  >>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
  >>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

  >>> article = "UN Chief says there is no military solution in Syria"
  >>> inputs = tokenizer(article, return_tensors="pt")

  >>> translated_tokens = model.generate(
  ...     **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30
  ... )
  >>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
  Le chef de l'ONU dit qu'il n'y a pas de solution militaire en Syrie
  ```

## NllbTokenizer

### class transformers.NllbTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb.py#L38)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' tokenizer\_file = None src\_lang = None tgt\_lang = None sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None additional\_special\_tokens = None legacy\_behaviour = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
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
* **tokenizer\_file** (`str`, *optional*) —
  The path to a tokenizer file to use instead of the vocab file.
* **src\_lang** (`str`, *optional*) —
  The language to use as source language for translation.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **sp\_model\_kwargs** (`dict[str, str]`) —
  Additional keyword arguments to pass to the model initialization.

Construct an NLLB tokenizer.

Adapted from [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on
[SentencePiece](https://github.com/google/sentencepiece).

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>

<tokens> <eos>` for target language documents.

Examples:


```
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb.py#L245)

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
adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:

* `input_ids` (for encoder) `X [eos, src_lang_code]`
* `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.

## NllbTokenizerFast

### class transformers.NllbTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb_fast.py#L42)

( vocab\_file = None tokenizer\_file = None bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' src\_lang = None tgt\_lang = None additional\_special\_tokens = None legacy\_behaviour = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
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
* **tokenizer\_file** (`str`, *optional*) —
  The path to a tokenizer file to use instead of the vocab file.
* **src\_lang** (`str`, *optional*) —
  The language to use as source language for translation.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.

Construct a “fast” NLLB tokenizer (backed by HuggingFace’s *tokenizers* library). Based on
[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>

<tokens> <eos>` for target language documents.

Examples:


```
>>> from transformers import NllbTokenizerFast

>>> tokenizer = NllbTokenizerFast.from_pretrained(
...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb_fast.py#L178)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. The special tokens depend on calling set\_lang.

An NLLB sequence has the following format, where `X` represents the sequence:

* `input_ids` (for encoder) `X [eos, src_lang_code]`
* `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb_fast.py#L207)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not
make use of token type ids, therefore a list of zeros is returned.

#### set\_src\_lang\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb_fast.py#L262)

( src\_lang  )

Reset the special tokens to the source lang setting.

* In legacy mode: No prefix and suffix=[eos, src\_lang\_code].
* In default mode: Prefix=[src\_lang\_code], suffix = [eos]

#### set\_tgt\_lang\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nllb/tokenization_nllb_fast.py#L285)

( lang: str  )

Reset the special tokens to the target lang setting.

* In legacy mode: No prefix and suffix=[eos, tgt\_lang\_code].
* In default mode: Prefix=[tgt\_lang\_code], suffix = [eos]

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nllb.md)
