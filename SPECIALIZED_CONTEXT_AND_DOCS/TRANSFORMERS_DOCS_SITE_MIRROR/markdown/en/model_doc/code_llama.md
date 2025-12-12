# CodeLlama

[Code Llama](https://huggingface.co/papers/2308.12950) is a specialized family of large language models based on [Llama 2](./llama2) for coding tasks.  It comes in different flavors - general code, Python-specific, and instruction-following variant - all available in 7B, 13B, 34B, and 70B parameters. Code Llama models can generate, explain, and even fill in missing parts of your code (called "infilling"). It can also handle very long contexts with stable generation up to 100k tokens, even though it was trained on sequences of 16K tokens.

You can find all the original Code Llama checkpoints under the [Code Llama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933) collection.

> [!TIP]
> Click on the Code Llama models in the right sidebar for more examples of how to apply Code Llama to different coding tasks.

The example below demonstrates how to generate code with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and from the command line.

```py
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
infill_result = pipe("def remove_non_ascii(s: str) -> str:\n    \"\"\" \n    return result", max_new_tokens=200)
print(infill_result[0]['generated_text'])
```

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/CodeLlama-7b-hf",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# basic code generation
prompt = "# Function to calculate the factorial of a number\ndef factorial(n):"
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **input_ids,
    max_new_tokens=256,
    cache_implementation="static"
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# infilling
infill_prompt = "def remove_non_ascii(s: str) -> str:\n    \"\"\" \n    return result"
input_ids = tokenizer(infill_prompt, return_tensors="pt").to(model.device)

filled_output = model.generate(**input_ids, max_new_tokens=200)
filled_text = tokenizer.decode(filled_output[0], skip_special_tokens=True)
print(filled_text)
```

```bash
echo -e "# Function to calculate the factorial of a number\ndef factorial(n):" | transformers run --task text-generation --model meta-llama/CodeLlama-7b-hf --device 0
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
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

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("meta-llama/CodeLlama-7b-hf")
visualizer("""def func(a, b):
  return a + b""")
```

    

## Notes

- Infilling is only available in the 7B and 13B base models, and not in the Python, Instruct, 34B, or 70B models.
- Use the `` token where you want your input to be filled. The tokenizer splits this token to create a formatted input string that follows the [original training pattern](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402). This is more robust than preparing the pattern yourself.

    ```py
    from transformers import LlamaForCausalLM, CodeLlamaTokenizer

    tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
    PROMPT = '''def remove_non_ascii(s: str) -> str:
        """ 
        return result
    '''
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids, max_new_tokens=128)

    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print(PROMPT.replace("", filling))
    ```

- Use `bfloat16` for further training or fine-tuning and `float16` for inference.
- The `BOS` character is not used for infilling when encoding the prefix or suffix, but only at the beginning of each prompt.
- The tokenizer is a byte-pair encoding model based on [SentencePiece](https://github.com/google/sentencepiece). During decoding, if the first token is the start of the word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.

## CodeLlamaTokenizer[[transformers.CodeLlamaTokenizer]]

#### transformers.CodeLlamaTokenizer[[transformers.CodeLlamaTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/code_llama/tokenization_code_llama.py#L44)

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.

```python
>>> from transformers import CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
>>> tokenizer.encode("Hello this is a test")
[1, 15043, 445, 338, 263, 1243]
```

If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
[post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods. The default configuration match that of
[meta-llama/CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
which supports prompt infilling.

get_special_tokens_masktransformers.CodeLlamaTokenizer.get_special_tokens_maskhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1335[{"name": "token_ids_0", "val": ": list[int]"}, {"name": "token_ids_1", "val": ": Optional[list[int]] = None"}, {"name": "already_has_special_tokens", "val": ": bool = False"}]- **token_ids_0** -- List of IDs for the (possibly already formatted) sequence.
- **token_ids_1** -- Unused when `already_has_special_tokens=True`. Must be None in that case.
- **already_has_special_tokens** -- Whether the sequence is already formatted with special tokens.0A list of integers in the range [0, 1]1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with `already_has_special_tokens=True` to build a mask over an
already-formatted sequence. In that case, we compute the mask by checking membership in `all_special_ids`.

**Parameters:**

clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`) : Whether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra spaces.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

prefix_token (`str`, *optional*, defaults to `"▁"`) : Prefix token used for infilling.

middle_token (`str`, *optional*, defaults to `"▁"`) : Middle token used for infilling.

suffix_token (`str`, *optional*, defaults to `"▁"`) : Suffix token used for infilling.

eot_token (`str`, *optional*, defaults to `"▁"`) : End of text token used for infilling.

fill_token (`str`, *optional*, defaults to `""`) : The token used to split the input between the prefix and suffix.

additional_special_tokens (`list[str]`, *optional*) : Additional special tokens used by the tokenizer.

add_bos_token (`bool`, *optional*, defaults to `True`) : Whether to add a beginning of sequence token at the start of sequences.

add_eos_token (`bool`, *optional*, defaults to `False`) : Whether to add an end of sequence token at the end of sequences.

use_default_system_prompt (`bool`, *optional*, defaults to `False`) : Whether or not the default system prompt for Llama should be used.

add_prefix_space (`bool`, *optional*) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.

merges (`str` or `list`, *optional*) : Custom merges list. If not provided, merges are loaded from merges_file.

vocab_file (`str`, *optional*) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that contains the vocabulary necessary to instantiate a tokenizer.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### save_vocabulary[[transformers.CodeLlamaTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L384)

## CodeLlamaTokenizerFast[[transformers.CodeLlamaTokenizer]]

#### transformers.CodeLlamaTokenizer[[transformers.CodeLlamaTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/code_llama/tokenization_code_llama.py#L44)

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.

```python
>>> from transformers import CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
>>> tokenizer.encode("Hello this is a test")
[1, 15043, 445, 338, 263, 1243]
```

If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
[post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods. The default configuration match that of
[meta-llama/CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
which supports prompt infilling.

get_special_tokens_masktransformers.CodeLlamaTokenizer.get_special_tokens_maskhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1335[{"name": "token_ids_0", "val": ": list[int]"}, {"name": "token_ids_1", "val": ": Optional[list[int]] = None"}, {"name": "already_has_special_tokens", "val": ": bool = False"}]- **token_ids_0** -- List of IDs for the (possibly already formatted) sequence.
- **token_ids_1** -- Unused when `already_has_special_tokens=True`. Must be None in that case.
- **already_has_special_tokens** -- Whether the sequence is already formatted with special tokens.0A list of integers in the range [0, 1]1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with `already_has_special_tokens=True` to build a mask over an
already-formatted sequence. In that case, we compute the mask by checking membership in `all_special_ids`.

**Parameters:**

clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`) : Whether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra spaces.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

prefix_token (`str`, *optional*, defaults to `"▁"`) : Prefix token used for infilling.

middle_token (`str`, *optional*, defaults to `"▁"`) : Middle token used for infilling.

suffix_token (`str`, *optional*, defaults to `"▁"`) : Suffix token used for infilling.

eot_token (`str`, *optional*, defaults to `"▁"`) : End of text token used for infilling.

fill_token (`str`, *optional*, defaults to `""`) : The token used to split the input between the prefix and suffix.

additional_special_tokens (`list[str]`, *optional*) : Additional special tokens used by the tokenizer.

add_bos_token (`bool`, *optional*, defaults to `True`) : Whether to add a beginning of sequence token at the start of sequences.

add_eos_token (`bool`, *optional*, defaults to `False`) : Whether to add an end of sequence token at the end of sequences.

use_default_system_prompt (`bool`, *optional*, defaults to `False`) : Whether or not the default system prompt for Llama should be used.

add_prefix_space (`bool`, *optional*) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.

merges (`str` or `list`, *optional*) : Custom merges list. If not provided, merges are loaded from merges_file.

vocab_file (`str`, *optional*) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that contains the vocabulary necessary to instantiate a tokenizer.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### update_post_processor[[transformers.CodeLlamaTokenizer.update_post_processor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L397)

Updates the underlying post processor with the current `bos_token` and `eos_token`.
#### save_vocabulary[[transformers.CodeLlamaTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L384)
