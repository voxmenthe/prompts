# NLLB

    
        
        
        
    

## Overview

[NLLB: No Language Left Behind](https://huggingface.co/papers/2207.04672) is a multilingual translation model. It's trained on data using data mining techniques tailored for low-resource languages and supports over 200 languages. NLLB features a conditional compute architecture using a Sparsely Gated Mixture of Experts.

You can find all the original NLLB checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=nllb) organization.

> [!TIP]
> This model was contributed by [Lysandre](https://huggingface.co/lysandre).
> Click on the NLLB models in the right sidebar for more examples of how to apply NLLB to different translation tasks.

The example below demonstrates how to translate text with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
import torch
from transformers import pipeline

pipeline = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn", dtype=torch.float16, device=0)
pipeline("UN Chief says there is no military solution in Syria")
```

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", dtype="auto", attn_implementation="sdpa")

article = "UN Chief says there is no military solution in Syria"
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"), max_length=30
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
```

```bash
echo -e "UN Chief says there is no military solution in Syria" | transformers run --task "translation_en_to_fr" --model facebook/nllb-200-distilled-600M --device 0
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 8-bits.

```python
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

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("facebook/nllb-200-distilled-600M")
visualizer("UN Chief says there is no military solution in Syria")
```

    

## Notes

- The tokenizer was updated in April 2023 to prefix the source sequence with the source language rather than the target language. This prioritizes zero-shot performance at a minor cost to supervised performance.

   ```python
   >>> from transformers import NllbTokenizer

   >>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
   >>> tokenizer("How was your day?").input_ids
   [256047, 13374, 1398, 4260, 4039, 248130, 2]
   ```

   To revert to the legacy behavior, use the code example below.

   ```python
   >>> from transformers import NllbTokenizer

   >>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
   ```

- For non-English languages, specify the language's [BCP-47](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) code with the `src_lang` keyword as shown below.

- See example below for a translation from Romanian to German.

    ```python
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

## NllbTokenizer[[transformers.NllbTokenizer]]

#### transformers.NllbTokenizer[[transformers.NllbTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nllb/tokenization_nllb.py#L35)

Construct an NLLB tokenizer (backed by HuggingFace's *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

The tokenization method is `  ` for source language documents, and `
 ` for target language documents.

Examples:

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

set_src_lang_special_tokenstransformers.NllbTokenizer.set_src_lang_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/nllb/tokenization_nllb.py#L266[{"name": "src_lang", "val": ""}]
Reset the special tokens to the source lang setting.
- In legacy mode: No prefix and suffix=[eos, src_lang_code].
- In default mode: Prefix=[src_lang_code], suffix = [eos]

**Parameters:**

vocab_file (`str`, *optional*) : Path to the vocabulary file.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

sep_token (`str`, *optional*, defaults to `""`) : The separator token.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values.

src_lang (`str`, *optional*) : The language to use as source language for translation.

tgt_lang (`str`, *optional*) : The language to use as target language for translation.

legacy_behaviour (`bool`, *optional*, defaults to `False`) : Whether to use legacy behaviour (suffix pattern) or new behaviour (prefix pattern).
#### set_tgt_lang_special_tokens[[transformers.NllbTokenizer.set_tgt_lang_special_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nllb/tokenization_nllb.py#L289)

Reset the special tokens to the target lang setting.
- In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
- In default mode: Prefix=[tgt_lang_code], suffix = [eos]

## NllbTokenizerFast[[transformers.NllbTokenizer]]

#### transformers.NllbTokenizer[[transformers.NllbTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nllb/tokenization_nllb.py#L35)

Construct an NLLB tokenizer (backed by HuggingFace's *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

The tokenization method is `  ` for source language documents, and `
 ` for target language documents.

Examples:

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

set_src_lang_special_tokenstransformers.NllbTokenizer.set_src_lang_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/nllb/tokenization_nllb.py#L266[{"name": "src_lang", "val": ""}]
Reset the special tokens to the source lang setting.
- In legacy mode: No prefix and suffix=[eos, src_lang_code].
- In default mode: Prefix=[src_lang_code], suffix = [eos]

**Parameters:**

vocab_file (`str`, *optional*) : Path to the vocabulary file.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

sep_token (`str`, *optional*, defaults to `""`) : The separator token.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values.

src_lang (`str`, *optional*) : The language to use as source language for translation.

tgt_lang (`str`, *optional*) : The language to use as target language for translation.

legacy_behaviour (`bool`, *optional*, defaults to `False`) : Whether to use legacy behaviour (suffix pattern) or new behaviour (prefix pattern).
#### set_tgt_lang_special_tokens[[transformers.NllbTokenizer.set_tgt_lang_special_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nllb/tokenization_nllb.py#L289)

Reset the special tokens to the target lang setting.
- In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
- In default mode: Prefix=[tgt_lang_code], suffix = [eos]
