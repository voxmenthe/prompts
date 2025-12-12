*This model was released on 2021-05-28 and added to Hugging Face Transformers on 2021-06-01.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# ByT5

[ByT5](https://huggingface.co/papers/2105.13626) is tokenizer-free version of the [T5](./t5) model designed to works directly on raw UTF-8 bytes. This means it can process any language, more robust to noise like typos, and simpler to use because it doesn’t require a preprocessing pipeline.

You can find all the original ByT5 checkpoints under the [Google](https://huggingface.co/google?search_models=byt5) organization.

Refer to the [T5](./t5) docs for more examples of how to apply ByT5 to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) and from the command line.

Pipeline

AutoModel

transformers-cli


```
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

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
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

* It is recommended to use the tokenizer for batched inference and training.
* The example below shows how to use the model without a tokenizer.


  ```
  import torch
  from transformers import AutoModelForSeq2SeqLM

  model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

  num_special_tokens = 3

  input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens
  labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens
  loss = model(input_ids, labels=labels).loss
  loss.item()
  ```
* ByT5 uses the top byte values (258, 257, etc.) for masking instead of sentinel tokens like `{extra_id_0}`.


  ```
  # Example: character-level denoising with mask tokens
  input_ids = tokenizer("The dog chases a ball in the park.").input_ids
  masked_input = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
  output = model.generate(masked_input, max_length=100)
  ```

## ByT5Tokenizer

### class transformers.ByT5Tokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/byt5/tokenization_byt5.py#L27)

( eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' extra\_ids = 125 additional\_special\_tokens = None \*\*kwargs  )

Parameters

* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **extra\_ids** (`int`, *optional*, defaults to 125) —
  Add a number of extra ids added to the end of the vocabulary for use as sentinels. These tokens are
  accessible as “id{%d}>” where ”{%d}” is a number between 0 and extra\_ids-1. Extra tokens are
  indexed from the end of the vocabulary up to beginning (“” is the last token in the vocabulary
  like in ByT5 preprocessing see
  [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)).
* **additional\_special\_tokens** (`list[str]`, *optional*) —
  Additional special tokens used by the tokenizer.

Construct a ByT5 tokenizer. ByT5 simply uses raw bytes utf-8 encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/byt5/tokenization_byt5.py#L171)

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

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/byt5/tokenization_byt5.py#L217)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/byt5/tokenization_byt5.py#L149)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/byt5/tokenization_byt5.py#L110)

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/byt5.md)
