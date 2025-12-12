# Data Collator

Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of
the same type as the elements of `train_dataset` or `eval_dataset`.

To be able to build batches, data collators may apply some processing (like padding). Some of them (like
[DataCollatorForLanguageModeling](/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling)) also apply some random data augmentation (like random masking)
on the formed batch.

Examples of use can be found in the [example scripts](../examples) or [example notebooks](../notebooks).

## Default data collator[[transformers.default_data_collator]]

#### transformers.default_data_collator[[transformers.default_data_collator]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L71)

Very simple data collator that simply collates batches of dict-like objects and performs special handling for
potential keys named:

- `label`: handles a single value (int or float) per object
- `label_ids`: handles a list of values per object

Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
to the model. See glue and ner for example of how it's useful.

## DefaultDataCollator[[transformers.DefaultDataCollator]]

#### transformers.DefaultDataCollator[[transformers.DefaultDataCollator]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L95)

Very simple data collator that simply collates batches of dict-like objects and performs special handling for
potential keys named:

- `label`: handles a single value (int or float) per object
- `label_ids`: handles a list of values per object

Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
to the model. See glue and ner for example of how it's useful.

This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
helpful if you need to set a return_tensors value at initialization.

**Parameters:**

return_tensors (`str`, *optional*, defaults to `"pt"`) : The type of Tensor to return. Allowable values are "np", or "pt".

## DataCollatorWithPadding[[transformers.DataCollatorWithPadding]]

#### transformers.DataCollatorWithPadding[[transformers.DataCollatorWithPadding]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L191)

Data collator that will dynamically pad the inputs received.

**Parameters:**

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) or [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)) : The tokenizer used for encoding the data.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).

max_length (`int`, *optional*) : Maximum length of the returned list and optionally padding length (see above).

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value.  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.0 (Volta).

return_tensors (`str`, *optional*, defaults to `"pt"`) : The type of Tensor to return. Allowable values are "np", or "pt".

## DataCollatorForTokenClassification[[transformers.DataCollatorForTokenClassification]]

#### transformers.DataCollatorForTokenClassification[[transformers.DataCollatorForTokenClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L243)

Data collator that will dynamically pad the inputs received, as well as the labels.

**Parameters:**

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) or [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)) : The tokenizer used for encoding the data.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).

max_length (`int`, *optional*) : Maximum length of the returned list and optionally padding length (see above).

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value.  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.0 (Volta).

label_pad_token_id (`int`, *optional*, defaults to -100) : The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).

return_tensors (`str`, *optional*, defaults to `"pt"`) : The type of Tensor to return. Allowable values are "np", or "pt".

## DataCollatorForSeq2Seq[[transformers.DataCollatorForSeq2Seq]]

#### transformers.DataCollatorForSeq2Seq[[transformers.DataCollatorForSeq2Seq]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L487)

Data collator that will dynamically pad the inputs received, as well as the labels.

**Parameters:**

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) or [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)) : The tokenizer used for encoding the data.

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel), *optional*) : The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to prepare the *decoder_input_ids*  This is useful when using *label_smoothing* to avoid calculating loss twice.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).

max_length (`int`, *optional*) : Maximum length of the returned list and optionally padding length (see above).

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value.  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.0 (Volta).

label_pad_token_id (`int`, *optional*, defaults to -100) : The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).

return_tensors (`str`, *optional*, defaults to `"pt"`) : The type of Tensor to return. Allowable values are "np", or "pt".

## DataCollatorForLanguageModeling[[transformers.DataCollatorForLanguageModeling]]

#### transformers.DataCollatorForLanguageModeling[[transformers.DataCollatorForLanguageModeling]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L619)

Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
are not all of the same length.

For best performance, this data collator should be used with a dataset having items that are dictionaries or
BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) or a
[PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) with the argument `return_special_tokens_mask=True`.

1. Default Behavior:
   - `mask_replace_prob=0.8`, `random_replace_prob=0.1`.
   - Expect 80% of masked tokens replaced with `[MASK]`, 10% replaced with random tokens, and 10% left unchanged.

2. All masked tokens replaced by `[MASK]`:
   - `mask_replace_prob=1.0`, `random_replace_prob=0.0`.
   - Expect all masked tokens to be replaced with `[MASK]`. No tokens are left unchanged or replaced with random tokens.

3. No `[MASK]` replacement, only random tokens:
   - `mask_replace_prob=0.0`, `random_replace_prob=1.0`.
   - Expect all masked tokens to be replaced with random tokens. No `[MASK]` replacements or unchanged tokens.

4. Balanced replacement:
   - `mask_replace_prob=0.5`, `random_replace_prob=0.4`.
   - Expect 50% of masked tokens replaced with `[MASK]`, 40% replaced with random tokens, and 10% left unchanged.

Note:
The sum of `mask_replace_prob` and `random_replace_prob` must not exceed 1. If their sum is less than 1, the
remaining proportion will consist of masked tokens left unchanged.

numpy_mask_tokenstransformers.DataCollatorForLanguageModeling.numpy_mask_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L890[{"name": "inputs", "val": ": typing.Any"}, {"name": "special_tokens_mask", "val": ": typing.Optional[typing.Any] = None"}, {"name": "offset_mapping", "val": ": typing.Optional[typing.Any] = None"}]

Prepare masked tokens inputs/labels for masked language modeling.

**Parameters:**

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) or [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)) : The tokenizer used for encoding the data.

mlm (`bool`, *optional*, defaults to `True`) : Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked tokens and the value to predict for the masked token.

whole_word_mask (`bool`, *optional*, defaults to `False`) : Whether or not to mask whole words instead of individual tokens.

mlm_probability (`float`, *optional*, defaults to 0.15) : The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.

mask_replace_prob (`float`, *optional*, defaults to 0.8) : The probability with which masked tokens are replaced by the tokenizer's mask token (e.g., `[MASK]`). Defaults to 0.8, meaning 80% of the masked tokens will be replaced with `[MASK]`. Only works when `mlm` is set to `True`.

random_replace_prob (`float`, *optional*, defaults to 0.1) : The probability with which masked tokens are replaced by random tokens from the tokenizer's vocabulary. Defaults to 0.1, meaning 10% of the masked tokens will be replaced with random tokens. The remaining masked tokens (1 - mask_replace_prob - random_replace_prob) are left unchanged. Only works when `mlm` is set to `True`.

pad_to_multiple_of (`int`, *optional*) : If set, will pad the sequence to a multiple of the provided value.

return_tensors (`str`) : The type of Tensor to return. Allowable values are "np", or "pt".

seed (`int`, *optional*) : The seed to use for the random number generator for masking. If not provided, the global RNG will be used.
#### torch_mask_tokens[[transformers.DataCollatorForLanguageModeling.torch_mask_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L796)

Prepare masked tokens inputs/labels for masked language modeling.

## DataCollatorForWholeWordMask[[transformers.DataCollatorForWholeWordMask]]

#### transformers.DataCollatorForWholeWordMask[[transformers.DataCollatorForWholeWordMask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L1019)

Data collator used for language modeling that masks entire words.

- collates batches of tensors, honoring their tokenizer's pad_token
- preprocesses batches for masked language modeling

numpy_mask_tokenstransformers.DataCollatorForWholeWordMask.numpy_mask_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L890[{"name": "inputs", "val": ": typing.Any"}, {"name": "special_tokens_mask", "val": ": typing.Optional[typing.Any] = None"}, {"name": "offset_mapping", "val": ": typing.Optional[typing.Any] = None"}]

Prepare masked tokens inputs/labels for masked language modeling.
#### torch_mask_tokens[[transformers.DataCollatorForWholeWordMask.torch_mask_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L796)

Prepare masked tokens inputs/labels for masked language modeling.

## DataCollatorForPermutationLanguageModeling[[transformers.DataCollatorForPermutationLanguageModeling]]

#### transformers.DataCollatorForPermutationLanguageModeling[[transformers.DataCollatorForPermutationLanguageModeling]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L1139)

Data collator used for permutation language modeling.

- collates batches of tensors, honoring their tokenizer's pad_token
- preprocesses batches for permutation language modeling with procedures specific to XLNet

numpy_mask_tokenstransformers.DataCollatorForPermutationLanguageModeling.numpy_mask_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L1265[{"name": "inputs", "val": ": typing.Any"}]

The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

0. Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
1. Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
2. Reserve a context of length `context_length = span_length / plm_probability` to surround span to be
   masked
3. Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length -
   span_length]` and mask tokens `start_index:start_index + span_length`
4. Set `cur_len = cur_len + context_length`. If `cur_len = 7.5 (Volta).

return_tensors (`str`, *optional*, defaults to `"pt"`) : The type of Tensor to return. Allowable values are "np", or "pt".
