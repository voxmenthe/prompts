# ByT5

[ByT5](https://huggingface.co/papers/2105.13626) is tokenizer-free version of the [T5](./t5) model designed to works directly on raw UTF-8 bytes. This means it can process any language, more robust to noise like typos, and simpler to use because it doesn't require a preprocessing pipeline.

You can find all the original ByT5 checkpoints under the [Google](https://huggingface.co/google?search_models=byt5) organization.

> [!TIP]
> Refer to the [T5](./t5) docs for more examples of how to apply ByT5 to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) and from the command line.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="google/byt5-small",
    dtype=torch.float16,
    device=0
)
pipeline("translate English to French: The weather is nice today")
```

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/byt5-small"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-small",
    dtype=torch.float16,
    device_map="auto"
)

input_ids = tokenizer("summarize: Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```bash
echo -e "translate English to French: Life is beautiful." | transformers run --task text2text-generation --model google/byt5-small --device 0
```

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```python
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-xl",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl")
input_ids = tokenizer("translate English to French: The weather is nice today.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- It is recommended to use the tokenizer for batched inference and training.
- The example below shows how to use the model without a tokenizer.

    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

    num_special_tokens = 3

    input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens
    labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens
    loss = model(input_ids, labels=labels).loss
    loss.item()
    ```

- ByT5 uses the top byte values (258, 257, etc.) for masking instead of sentinel tokens like `{extra_id_0}`.

    ```python
    # Example: character-level denoising with mask tokens
    input_ids = tokenizer("The dog chases a ball in the park.").input_ids
    masked_input = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
    output = model.generate(masked_input, max_length=100)
    ```

## ByT5Tokenizer[[transformers.ByT5Tokenizer]]

#### transformers.ByT5Tokenizer[[transformers.ByT5Tokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/byt5/tokenization_byt5.py#L27)

Construct a ByT5 tokenizer. ByT5 simply uses raw bytes utf-8 encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

build_inputs_with_special_tokenstransformers.ByT5Tokenizer.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/byt5/tokenization_byt5.py#L171[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`list[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`list[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`list[int]`List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A sequence has the following format:

- single sequence: `X `
- pair of sequences: `A  B `

**Parameters:**

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

extra_ids (`int`, *optional*, defaults to 125) : Add a number of extra ids added to the end of the vocabulary for use as sentinels. These tokens are accessible as "" where "{%d}" is a number between 0 and extra_ids-1. Extra tokens are indexed from the end of the vocabulary up to beginning ("" is the last token in the vocabulary like in ByT5 preprocessing see [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)).

additional_special_tokens (`list[str]`, *optional*) : Additional special tokens used by the tokenizer.

**Returns:**

``list[int]``

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
#### convert_tokens_to_string[[transformers.ByT5Tokenizer.convert_tokens_to_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/byt5/tokenization_byt5.py#L217)

Converts a sequence of tokens (string) in a single string.
#### create_token_type_ids_from_sequences[[transformers.ByT5Tokenizer.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/byt5/tokenization_byt5.py#L149)

Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
make use of token type ids, therefore a list of zeros is returned.

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of zeros.
#### get_special_tokens_mask[[transformers.ByT5Tokenizer.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/byt5/tokenization_byt5.py#L110)

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

``list[int]``

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
