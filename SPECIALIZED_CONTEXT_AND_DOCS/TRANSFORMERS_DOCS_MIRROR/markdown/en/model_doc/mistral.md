*This model was released on 2023-10-10 and added to Hugging Face Transformers on 2023-09-27.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white) ![Tensor parallelism](https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white)

# Mistral

[Mistral](https://huggingface.co/papers/2310.06825) is a 7B parameter language model, available as a pretrained and instruction-tuned variant, focused on balancing
the scaling costs of large models with performance and efficient inference. This model uses sliding window attention (SWA) trained with a 8K context length and a fixed cache size to handle longer sequences more effectively. Grouped-query attention (GQA) speeds up inference and reduces memory requirements. Mistral also features a byte-fallback BPE tokenizer to improve token handling and efficiency by ensuring characters are never mapped to out-of-vocabulary tokens.

You can find all the original Mistral checkpoints under the [Mistral AI\_](https://huggingface.co/mistralai) organization.

Click on the Mistral models in the right sidebar for more examples of how to apply Mistral to different language tasks.

The example below demonstrates how to chat with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
>>> import torch
>>> from transformers import pipeline

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", dtype=torch.bfloat16, device=0)
>>> chatbot(messages)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.


```
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

>>> # specify how to quantize the model
>>> quantization_config = BitsAndBytesConfig(
...         load_in_4bit=True,
...         bnb_4bit_quant_type="nf4",
...         bnb_4bit_compute_dtype="torch.float16",
... )

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantization_config=True, dtype=torch.bfloat16, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

>>> prompt = "My favourite condiment is"

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.


```
>>> from transformers.utils.attention_visualizer import AttentionMaskVisualizer

>>> visualizer = AttentionMaskVisualizer("mistralai/Mistral-7B-Instruct-v0.3")
>>> visualizer("Do you have mayonnaise recipes?")
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mistral-attn-mask.png)

## MistralConfig

### class transformers.MistralConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/configuration_mistral.py#L24)

( vocab\_size = 32000 hidden\_size = 4096 intermediate\_size = 14336 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = 8 head\_dim = None hidden\_act = 'silu' max\_position\_embeddings = 131072 initializer\_range = 0.02 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = None bos\_token\_id = 1 eos\_token\_id = 2 tie\_word\_embeddings = False rope\_theta = 10000.0 sliding\_window = 4096 attention\_dropout = 0.0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the Mistral model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [MistralModel](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 14336) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
* **head\_dim** (`int`, *optional*, defaults to `hidden_size // num_attention_heads`) —
  The attention head dimension.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to `4096*32`) —
  The maximum sequence length that this model might ever be used with. Mistral’s sliding window attention
  allows sequence of up to 4096\*32 tokens.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*) —
  The id of the padding token.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the “beginning-of-sequence” token.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the “end-of-sequence” token.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **sliding\_window** (`int`, *optional*, defaults to 4096) —
  Sliding window attention window size. If not specified, will default to `4096`.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.

This is the configuration class to store the configuration of a [MistralModel](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel). It is used to instantiate an
Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import MistralModel, MistralConfig

>>> # Initializing a Mistral 7B style configuration
>>> configuration = MistralConfig()

>>> # Initializing a model from the Mistral 7B style configuration
>>> model = MistralModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MistralCommonTokenizer

### class transformers.MistralCommonTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L158)

( tokenizer\_path: typing.Union[str, os.PathLike, pathlib.Path] mode: ValidationMode = <ValidationMode.test: 'test'> model\_max\_length: int = 1000000000000000019884624838656 padding\_side: str = 'left' truncation\_side: str = 'right' model\_input\_names: typing.Optional[list[str]] = None clean\_up\_tokenization\_spaces: bool = False \*\*kwargs  )

Class to wrap `mistral-common` tokenizers.

`mistral-common` is the official tokenizer library for Mistral AI models. To use it, you need to install it with:


```
pip install transformers[mistral-common]
```

Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.

For more info on `mistral-common`, see [mistral-common](https://github.com/mistralai/mistral-common).

This class is a wrapper around a `mistral_common.tokens.tokenizers.mistral.MistralTokenizer`.
It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer.

Supports the following methods from the `PreTrainedTokenizerBase` class:

* [get\_vocab()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.get_vocab): Returns the vocabulary as a dictionary of token to index.
* [encode()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.encode): Encode a string to a list of integers.
* [decode()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.decode): Decode a list of integers to a string.
* [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.batch_decode): Decode a batch of list of integers to a list of strings.
* [convert\_tokens\_to\_ids()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.convert_tokens_to_ids): Convert a list of tokens to a list of integers.
* [convert\_ids\_to\_tokens()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.convert_ids_to_tokens): Convert a list of integers to a list of tokens.
* [tokenize()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.tokenize): Tokenize a string.
* [get\_special\_tokens\_mask()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.get_special_tokens_mask): Get the special tokens mask for a list of tokens.
* [prepare\_for\_model()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.prepare_for_model): Prepare a list of inputs for the model.
* [pad()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.pad): Pad a list of inputs to the same length.
* [truncate\_sequences()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.truncate_sequences): Truncate a list of sequences to the same length.
* [apply\_chat\_template()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.apply_chat_template): Apply a chat template to a list of messages.
* `__call__()`: Tokenize a string or a list of strings.
* [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.from_pretrained): Download and cache a pretrained tokenizer from the Hugging Face model hub or local directory.
* [save\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.save_pretrained): Save a tokenizer to a directory, so it can be reloaded using the `from_pretrained` class method.
* [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub): Upload tokenizer to the Hugging Face model hub.

Here are the key differences with the `PreTrainedTokenizerBase` class:

* Pair of sequences are not supported. The signature have been kept for compatibility but all arguments related to pair of sequences are ignored. The return values of pairs are returned as `None`.
* The `is_split_into_words` argument is not supported.
* The `return_token_type_ids` argument is not supported.
* It is not possible to add new tokens to the tokenizer. Also the special tokens are handled differently from Transformers. In `mistral-common`, special tokens are never encoded directly. This means that: `tokenizer.encode("<s>")` will not return the ID of the `<s>` token. Instead, it will return a list of IDs corresponding to the tokenization of the string `"<s>"`. For more information, see the [mistral-common documentation](https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens).

If you have suggestions to improve this class, please open an issue on the [mistral-common GitHub repository](https://github.com/mistralai/mistral-common/issues) if it is related to the tokenizer or on the [Transformers GitHub repository](https://github.com/huggingface/transformers/issues) if it is related to the Hugging Face interface.

#### apply\_chat\_template

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1368)

( conversation: typing.Union[list[dict[str, str]], list[list[dict[str, str]]]] tools: typing.Optional[list[typing.Union[dict, typing.Callable]]] = None continue\_final\_message: bool = False tokenize: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: bool = False max\_length: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_dict: bool = False \*\*kwargs  ) → `Union[str, List[int], List[str], List[List[int]], BatchEncoding]`

Parameters

* **conversation** (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) — A list of dicts
  with “role” and “content” keys, representing the chat history so far.
* **tools** (`List[Union[Dict, Callable]]`, *optional*) —
  A list of tools (callable functions) that will be accessible to the model. If the template does not
  support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
  giving the name, description and argument types for the tool. See our
  [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
  for more information.
* **continue\_final\_message** (bool, *optional*) —
  If this is set, the chat will be formatted so that the final
  message in the chat is open-ended, without any EOS tokens. The model will continue this message
  rather than starting a new one. This allows you to “prefill” part of
  the model’s response for it. Cannot be used at the same time as `add_generation_prompt`.
* **tokenize** (`bool`, defaults to `True`) —
  Whether to tokenize the output. If `False`, the output will be a string.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, defaults to `False`) —
  Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
* **max\_length** (`int`, *optional*) —
  Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
  not specified, the tokenizer’s `max_length` attribute will be used as a default.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
  values are:
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
* **return\_dict** (`bool`, defaults to `False`) —
  Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
  If at least one conversation contains an image, its pixel values will be returned in the `pixel_values` key.
* **kwargs** (additional keyword arguments, *optional*) —
  Not supported by `MistralCommonTokenizer.apply_chat_template`.
  Will raise an error if used.

Returns

`Union[str, List[int], List[str], List[List[int]], BatchEncoding]`

A list of token ids representing the tokenized chat so far, including control
tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.

Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
ids.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L476)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `List[str]`

Parameters

* **sequences** (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Not supported by `MistralCommonTokenizer.batch_decode`.
  Will raise an error if used.

Returns

`List[str]`

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### convert\_ids\_to\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L523)

( ids: typing.Union[int, list[int]] skip\_special\_tokens: bool = False  ) → `str` or `List[str]`

Parameters

* **ids** (`int` or `List[int]`) —
  The token id (or token ids) to convert to tokens.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.

Returns

`str` or `List[str]`

The decoded token(s).

Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
added tokens.

#### convert\_tokens\_to\_ids

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L571)

( tokens: typing.Union[str, list[str]]  ) → `int` or `List[int]`

Parameters

* **tokens** (`str` or `List[str]`) — One or several token(s) to convert to token id(s).

Returns

`int` or `List[int]`

The token id or list of token ids.

Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
vocabulary.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L434)

( token\_ids: typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `str`

Parameters

* **token\_ids** (`Union[int, List[int], np.ndarray, torch.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Not supported by `MistralCommonTokenizer.decode`.
  Will raise an error if used.

Returns

`str`

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

#### encode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L367)

( text: typing.Union[str, list[int]] text\_pair: None = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None verbose: bool = True \*\*kwargs  ) → `List[int]`, `torch.Tensor`

Parameters

* **text** (`str` or `List[int]`) —
  The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).
* **text\_pair** (`None`, *optional*) —
  Not supported by `MistralCommonTokenizer.encode`. Kept to match `PreTrainedTokenizerBase.encode` signature.
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided.
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
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
* \***\*kwargs** — Not supported by `MistralCommonTokenizer.encode`.
  Will raise an error if used.

Returns

`List[int]`, `torch.Tensor`

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1689)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] \*init\_inputs mode: ValidationMode = <ValidationMode.test: 'test'> cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' model\_max\_length: int = 1000000000000000019884624838656 padding\_side: str = 'left' truncation\_side: str = 'right' model\_input\_names: typing.Optional[list[str]] = None clean\_up\_tokenization\_spaces: bool = False \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing the tokenizer config, for instance saved
    using the `MistralCommonTokenizer.tokenization_mistral_common.save_pretrained` method, e.g.,
    `./my_model_directory/`.
* **mode** (`ValidationMode`, *optional*, defaults to `ValidationMode.test`) —
  Validation mode for the `MistralTokenizer` tokenizer.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
  exist.
* **token** (`str` or *bool*, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **local\_files\_only** (`bool`, *optional*, defaults to `False`) —
  Whether or not to only rely on local files and not to attempt to download any files.
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **padding\_side** (`str`, *optional*, defaults to `"left"`) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **truncation\_side** (`str`, *optional*, defaults to `"right"`) —
  The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
* **model\_input\_names** (`List[string]`, *optional*) —
  The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
  `"attention_mask"`). Default value is picked from the class attribute of the same name.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should cleanup the spaces that were added when splitting the input text during the
  tokenization process.
* **kwargs** (additional keyword arguments, *optional*) —
  Not supported by `MistralCommonTokenizer.from_pretrained`.
  Will raise an error if used.

Instantiate a `MistralCommonTokenizer` from a predefined
tokenizer.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L746)

( token\_ids\_0: list token\_ids\_1: None = None already\_has\_special\_tokens: bool = False  ) → A list of integers in the range [0, 1]

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of ids of the sequence.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Not supported by `MistralCommonTokenizer`. Kept to match the interface of `PreTrainedTokenizerBase`.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

#### get\_vocab

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L345)

( ) → `Dict[str, int]`

Returns

`Dict[str, int]`

The vocabulary.

Returns the vocabulary as a dictionary of token to index.

This is a lossy conversion. There may be multiple token ids that decode to the same
string due to partial UTF-8 byte sequences being converted to �.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1130)

( encoded\_inputs: typing.Union[transformers.tokenization\_utils\_base.BatchEncoding, list[transformers.tokenization\_utils\_base.BatchEncoding], dict[str, list[int]], dict[str, list[list[int]]], list[dict[str, list[int]]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True max\_length: typing.Optional[int] = None pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_attention\_mask: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None verbose: bool = True  )

Parameters

* **encoded\_inputs** ([BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding), list of [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding), `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`) —
  Tokenized inputs. Can represent one input ([BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) or `Dict[str, List[int]]`) or a batch of
  tokenized inputs (list of [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding), *Dict[str, List[List[int]]]* or *List[Dict[str,
  List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
  collate function.

  Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors), see
  the note above for the return type.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different
    lengths).
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.

Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
in the batch.

Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
`self.pad_token_id`).

If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors, the
result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.

#### prepare\_for\_model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L842)

( ids: list pair\_ids: None = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_length: bool = False verbose: bool = True prepend\_batch\_axis: bool = False \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **ids** (`List[int]`) —
  Tokenized input ids of the first sequence.
* **pair\_ids** (`None`, *optional*) —
  Not supported by `MistralCommonTokenizer`. Kept to match the interface of `PreTrainedTokenizerBase`.
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided.
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
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
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
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`)

Prepares a sequence of input id so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1816)

( save\_directory: typing.Union[str, os.PathLike, pathlib.Path] push\_to\_hub: bool = False token: typing.Union[bool, str, NoneType] = None commit\_message: typing.Optional[str] = None repo\_id: typing.Optional[str] = None private: typing.Optional[bool] = None repo\_url: typing.Optional[str] = None organization: typing.Optional[str] = None \*\*kwargs  ) → A tuple of `str`

Parameters

* **save\_directory** (`str` or `os.PathLike`) — The path to a directory where the tokenizer will be saved.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **token** (`str` or *bool*, *optional*, defaults to `None`) —
  The token to use to push to the model hub. If `True`, will use the token in the `HF_TOKEN` environment
  variable.
* **commit\_message** (`str`, *optional*) — The commit message to use when pushing to the hub.
* **repo\_id** (`str`, *optional*) — The name of the repository to which push to the Hub.
* **private** (`bool`, *optional*) — Whether the model repository is private or not.
* **repo\_url** (`str`, *optional*) — The URL to the Git repository to which push to the Hub.
* **organization** (`str`, *optional*) — The name of the organization in which you would like to push your model.
* **kwargs** (`Dict[str, Any]`, *optional*) —
  Not supported by `MistralCommonTokenizer.save_pretrained`.
  Will raise an error if used.

Returns

A tuple of `str`

The files saved.

Save the full tokenizer state.

This method make sure the full tokenizer can then be re-loaded using the
`~MistralCommonTokenizer.tokenization_mistral_common.from_pretrained` class method.

#### tokenize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L606)

( text: str \*\*kwargs  ) → `List[str]`

Parameters

* **text** (`str`) —
  The sequence to be encoded.
* \***\*kwargs** (additional keyword arguments) —
  Not supported by `MistralCommonTokenizer.tokenize`.
  Will raise an error if used.

Returns

`List[str]`

The list of tokens.

Converts a string into a sequence of tokens, using the tokenizer.

Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.

#### truncate\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1293)

( ids: list pair\_ids: None = None num\_tokens\_to\_remove: int = 0 truncation\_strategy: typing.Union[str, transformers.tokenization\_utils\_base.TruncationStrategy] = 'longest\_first' stride: int = 0 \*\*kwargs  ) → `Tuple[List[int], None, List[int]]`

Parameters

* **ids** (`List[int]`) —
  Tokenized input ids. Can be obtained from a string by chaining the `tokenize` and
  `convert_tokens_to_ids` methods.
* **pair\_ids** (`None`, *optional*) —
  Not supported by `MistralCommonTokenizer`. Kept to match the signature of `PreTrainedTokenizerBase.truncate_sequences`.
* **num\_tokens\_to\_remove** (`int`, *optional*, defaults to 0) —
  Number of tokens to remove using the truncation strategy.
* **truncation\_strategy** (`str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `'longest_first'`) —
  The strategy to follow for truncation. Can be:
  + `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided.
  + `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
    than the model maximum admissible input size).
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a positive number, the overflowing tokens returned will contain some tokens from the main
  sequence returned. The value of this argument defines the number of additional tokens.

Returns

`Tuple[List[int], None, List[int]]`

The truncated `ids` and the list of
overflowing tokens. `None` is returned to match Transformers signature.

Truncates a sequence pair in-place following the strategy.

## MistralModel

### class transformers.MistralModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L307)

( config: MistralConfig  )

Parameters

* **config** ([MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Mistral Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L324)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MistralModel](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MistralForCausalLM

### class transformers.MistralForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L387)

( config  )

Parameters

* **config** ([MistralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mistral Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L401)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MistralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MistralForCausalLM

>>> model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## MistralForSequenceClassification

### class transformers.MistralForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L466)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForSequenceClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MistralForTokenClassification

### class transformers.MistralForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L462)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L254)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForTokenClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MistralForQuestionAnswering

### class transformers.MistralForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L470)

( config  )

* forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mistral.md)
