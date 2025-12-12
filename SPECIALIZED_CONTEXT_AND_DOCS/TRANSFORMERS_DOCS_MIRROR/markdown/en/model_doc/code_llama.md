*This model was released on 2023-08-24 and added to Hugging Face Transformers on 2023-08-25.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# CodeLlama

[Code Llama](https://huggingface.co/papers/2308.12950) is a specialized family of large language models based on [Llama 2](./llama2) for coding tasks. It comes in different flavors - general code, Python-specific, and instruction-following variant - all available in 7B, 13B, 34B, and 70B parameters. Code Llama models can generate, explain, and even fill in missing parts of your code (called “infilling”). It can also handle very long contexts with stable generation up to 100k tokens, even though it was trained on sequences of 16K tokens.

You can find all the original Code Llama checkpoints under the [Code Llama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933) collection.

Click on the Code Llama models in the right sidebar for more examples of how to apply Code Llama to different coding tasks.

The example below demonstrates how to generate code with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/CodeLlama-7b-hf",
    dtype=torch.float16,
    device_map=0
)

# basic code generation
result = pipe("# Function to calculate the factorial of a number\ndef factorial(n):", max_new_tokens=256)
print(result[0]['generated_text'])

# infilling
infill_result = pipe("def remove_non_ascii(s: str) -> str:\n    \"\"\" <FILL_ME>\n    return result", max_new_tokens=200)
print(infill_result[0]['generated_text'])
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.


```
# pip install bitsandbytes
import torch
from transformers import AutoModelForCausalLM, CodeLlamaTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-34b-hf")
model = AutoModelForCausalLM.from_pretrained(
   "meta-llama/CodeLlama-34b-hf",
   dtype=torch.bfloat16,
   device_map="auto",
   quantization_config=bnb_config
)

prompt = "# Write a Python function to check if a string is a palindrome\ndef is_palindrome(s):"
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=200, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.


```
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("meta-llama/CodeLlama-7b-hf")
visualizer("""def func(a, b):
  return a + b""")
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/codellama-attn-mask.png)

## Notes

* Infilling is only available in the 7B and 13B base models, and not in the Python, Instruct, 34B, or 70B models.
* Use the `<FILL_ME>` token where you want your input to be filled. The tokenizer splits this token to create a formatted input string that follows the [original training pattern](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402). This is more robust than preparing the pattern yourself.


  ```
  from transformers import LlamaForCausalLM, CodeLlamaTokenizer

  tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
  model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
  PROMPT = '''def remove_non_ascii(s: str) -> str:
      """ <FILL_ME>
      return result
  '''
  input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
  generated_ids = model.generate(input_ids, max_new_tokens=128)

  filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
  print(PROMPT.replace("<FILL_ME>", filling))
  ```
* Use `bfloat16` for further training or fine-tuning and `float16` for inference.
* The `BOS` character is not used for infilling when encoding the prefix or suffix, but only at the beginning of each prompt.
* The tokenizer is a byte-pair encoding model based on [SentencePiece](https://github.com/google/sentencepiece). During decoding, if the first token is the start of the word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.

## CodeLlamaTokenizer

### class transformers.CodeLlamaTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama.py#L51)

( vocab\_file unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' prefix\_token = '▁<PRE>' middle\_token = '▁<MID>' suffix\_token = '▁<SUF>' eot\_token = '▁<EOT>' fill\_token = '<FILL\_ME>' suffix\_first = False sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None add\_bos\_token = True add\_eos\_token = False clean\_up\_tokenization\_spaces = False additional\_special\_tokens = None use\_default\_system\_prompt = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **prefix\_token** (`str`, *optional*, defaults to `"▁<PRE>"`) —
  Prefix token used for infilling.
* **middle\_token** (`str`, *optional*, defaults to `"▁<MID>"`) —
  Middle token used for infilling.
* **suffix\_token** (`str`, *optional*, defaults to `"▁<SUF>"`) —
  Suffix token used for infilling.
* **eot\_token** (`str`, *optional*, defaults to `"▁<EOT>"`) —
  End of text token used for infilling.
* **fill\_token** (`str`, *optional*, defaults to `"<FILL_ME>"`) —
  The token used to split the input between the prefix and suffix.
* **suffix\_first** (`bool`, *optional*, defaults to `False`) —
  Whether the input prompt and suffix should be formatted with the suffix first.
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
* **add\_bos\_token** (`bool`, *optional*, defaults to `True`) —
  Whether to add a beginning of sequence token at the start of sequences.
* **add\_eos\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to add an end of sequence token at the end of sequences.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `False`) —
  Whether or not to clean up the tokenization spaces.
* **additional\_special\_tokens** (`list[str]`, *optional*) —
  Additional special tokens used by the tokenizer.
* **use\_default\_system\_prompt** (`bool`, *optional*, defaults to `False`) —
  Whether or not the default system prompt for Llama should be used.

Construct a CodeLlama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as
there is no padding token in the original model.

The default configuration match that of
[codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
which supports prompt infilling.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama.py#L359)

( token\_ids\_0 token\_ids\_1 = None  )

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama.py#L371)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama.py#L409)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of ids.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT

sequence pair mask has the following format:


```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

if token\_ids\_1 is None, only returns the first portion of the mask (0s).

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama.py#L331)

( save\_directory filename\_prefix: typing.Optional[str] = None  ) → `Tuple(str)`

Parameters

* **save\_directory** (`str`) —
  The directory in which to save the vocabulary.

Returns

`Tuple(str)`

Paths to the files saved.

Save the vocabulary and special tokens file to a directory.

## CodeLlamaTokenizerFast

### class transformers.CodeLlamaTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L49)

( vocab\_file = None tokenizer\_file = None clean\_up\_tokenization\_spaces = False unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' prefix\_token = '▁<PRE>' middle\_token = '▁<MID>' suffix\_token = '▁<SUF>' eot\_token = '▁<EOT>' fill\_token = '<FILL\_ME>' additional\_special\_tokens = None add\_bos\_token = True add\_eos\_token = False use\_default\_system\_prompt = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`, *optional*) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **tokenizer\_file** (`str`, *optional*) —
  [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
  contains everything needed to load the tokenizer.
* **clean\_up\_tokenization\_spaces** (`str`, *optional*, defaults to `False`) —
  Whether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
  spaces.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.
* **prefix\_token** (`str`, *optional*, defaults to `"▁<PRE>"`) —
  Prefix token used for infilling.
* **middle\_token** (`str`, *optional*, defaults to `"▁<MID>"`) —
  Middle token used for infilling.
* **suffix\_token** (`str`, *optional*, defaults to `"▁<SUF>"`) —
  Suffix token used for infilling.
* **eot\_token** (`str`, *optional*, defaults to `"▁<EOT>"`) —
  End of text token used for infilling.
* **fill\_token** (`str`, *optional*, defaults to `"<FILL_ME>"`) —
  The token used to split the input between the prefix and suffix.
* **additional\_special\_tokens** (`list[str]`, *optional*) —
  Additional special tokens used by the tokenizer.
* **add\_bos\_token** (`bool`, *optional*, defaults to `True`) —
  Whether to add a beginning of sequence token at the start of sequences.
* **add\_eos\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to add an end of sequence token at the end of sequences.
* **use\_default\_system\_prompt** (`bool`, *optional*, defaults to `False`) —
  Whether or not the default system prompt for Llama should be used.

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.


```
>>> from transformers import CodeLlamaTokenizerFast

>>> tokenizer = CodeLlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
>>> tokenizer.encode("Hello this is a test")
[1, 15043, 445, 338, 263, 1243]
```

If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
[post-processors] (<https://huggingface.co/docs/tokenizers/api/post-processors>) documentation.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods. The default configuration match that of
[meta-llama/CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
which supports prompt infilling.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L345)

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

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3913)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → A list of integers in the range [0, 1]

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of ids of the first sequence.
* **token\_ids\_1** (`list[int]`, *optional*) —
  List of ids of the second sequence.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) — The first tokenized sequence.
* **token\_ids\_1** (`list[int]`, *optional*) — The second tokenized sequence.

Returns

`list[int]`

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type
IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### update\_post\_processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L172)

( )

Updates the underlying post processor with the current `bos_token` and `eos_token`.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/code_llama/tokenization_code_llama_fast.py#L326)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/code_llama.md)
