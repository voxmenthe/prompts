*This model was released on 2024-09-17 and added to Hugging Face Transformers on 2024-09-14.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Pixtral

[Pixtral](https://huggingface.co/papers/2410.07073) is a multimodal model trained to understand natural images and documents. It accepts images in their natural resolution and aspect ratio without resizing or padding due to it’s 2D RoPE embeddings. In addition, Pixtral has a long 128K token context window for processing a large number of images. Pixtral couples a 400M vision encoder with a 12B Mistral Nemo decoder.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/pixtral_architecture.webp) Pixtral architecture. Taken from the [blog post.](https://mistral.ai/news/pixtral-12b/)

You can find all the original Pixtral checkpoints under the [Mistral AI](https://huggingface.co/mistralai/models?search=pixtral) organization.

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ).
Click on the Pixtral models in the right sidebar for more examples of how to apply Pixtral to different vision and language tasks.

AutoModel


```
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url_dog = "https://picsum.photos/id/237/200/300"
url_mountain = "https://picsum.photos/seed/picsum/200/300"

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can this animal"}, 
        {"type": "image", "url": url_dog}, 
        {"type": "text", "content": "live here?"}, 
        {"type": "image", "url" : url_mountain}
      ]
    }
]

inputs = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors"pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the model to 4-bits.


```
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

model_id = "mistral-community/pixtral-12b"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

dog_url = "https://picsum.photos/id/237/200/300"
mountain_url = "https://picsum.photos/seed/picsum/200/300"
dog_image = Image.open(requests.get(dog_url, stream=True).raw)
mountain_image = Image.open(requests.get(mountain_url, stream=True).raw)

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "text": "Can this animal"},
        {"type": "image"},
        {"type": "text", "text": "live here?"},
        {"type": "image"}
      ]
    }
]

prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=[dog_image, mountain_image], return_tensors="pt")

inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

generate_ids = model.generate(**inputs, max_new_tokens=100)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output)
```

## Notes

* Pixtral uses [PixtralVisionModel](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionModel) as the vision encoder and [MistralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM) for its language decoder.
* The model internally replaces `[IMG]` token placeholders with image embeddings.


  ```
  "<s>[INST][IMG]\nWhat are the things I should be cautious about when I visit this place?[/INST]"
  ```

  The `[IMG]` tokens are replaced with a number of `[IMG]` tokens that depend on the height and width of each image. Each row of the image is separated by a `[IMG_BREAK]` token and each image is separated by a `[IMG_END]` token. Use the `~Processor.apply_chat_template` method to handle these tokens for you.

## PixtralVisionConfig

### class transformers.PixtralVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/configuration_pixtral.py#L23)

( hidden\_size = 1024 intermediate\_size = 4096 num\_hidden\_layers = 24 num\_attention\_heads = 16 num\_channels = 3 image\_size = 1024 patch\_size = 16 hidden\_act = 'gelu' attention\_dropout = 0.0 rope\_theta = 10000.0 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of input channels in the input images.
* **image\_size** (`int`, *optional*, defaults to 1024) —
  Max dimension of the input images.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  Size of the image patches.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  Activation function used in the hidden layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  Dropout probability for the attention layers.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [PixtralVisionModel](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionModel). It is used to instantiate an
Pixtral vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to the vision encoder used by Pixtral-12B.

e.g. [pixtral-hf/pixtral-9b](https://huggingface.co/pixtral-hf/pixtral-9b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PixtralVisionModel, PixtralVisionConfig

>>> # Initializing a Pixtral-12B style configuration
>>> config = PixtralVisionConfig()

>>> # Initializing a model (with randomly initialized weights) from the configuration
>>> model = PixtralVisionModel(configuration)

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

## PixtralVisionModel

### class transformers.PixtralVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/modeling_pixtral.py#L446)

( config  )

Parameters

* **config** ([PixtralVisionModel](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pixtral Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/modeling_pixtral.py#L469)

( pixel\_values: Tensor image\_sizes: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*args \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PixtralImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor). See [PixtralImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([PixtralProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralProcessor) uses
  [PixtralImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor) for processing images).
* **image\_sizes** (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) —
  The sizes of the images in the batch, being (height, width) for each image.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PixtralVisionConfig](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PixtralVisionModel](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## PixtralImageProcessor

### class transformers.PixtralImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/image_processing_pixtral.py#L139)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None patch\_size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by
  `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"longest_edge" -- 1024}`):
  Size of the maximum dimension of either the height or width dimension of the image. Used to control how
  images are resized. If either the height or width are greater than `size["longest_edge"]` then both the height and width are rescaled by `height / ratio`, `width /ratio` where `ratio = max(height / longest_edge, width / longest_edge)`
* **patch\_size** (`dict[str, int]` *optional*, defaults to `{"height" -- 16, "width": 16}`):
  Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a Pixtral image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/image_processing_pixtral.py#L319)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None patch\_size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Describes the maximum input dimensions to the model.
* **patch\_size** (`dict[str, int]`, *optional*, defaults to `self.patch_size`) —
  Patch size in the model. Used to calculate the image after resizing.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## PixtralImageProcessorFast

### class transformers.PixtralImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/image_processing_pixtral_fast.py#L65)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.pixtral.image\_processing\_pixtral\_fast.PixtralFastImageProcessorKwargs]  )

Constructs a fast Pixtral image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/image_processing_pixtral_fast.py#L83)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.pixtral.image\_processing\_pixtral\_fast.PixtralFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>
* **patch\_size** (`dict[str, int]` *optional*, defaults to `{"height" -- 16, "width": 16}`):
  Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## PixtralProcessor

### class transformers.PixtralProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pixtral/processing_pixtral.py#L65)

( image\_processor = None tokenizer = None patch\_size: int = 16 spatial\_merge\_size: int = 1 chat\_template = None image\_token = '[IMG]' image\_break\_token = '[IMG\_BREAK]' image\_end\_token = '[IMG\_END]' \*\*kwargs  )

Parameters

* **image\_processor** ([PixtralImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  Patch size from the vision tower.
* **spatial\_merge\_size** (`int`, *optional*, defaults to 1) —
  The downsampling factor for the spatial merge operation.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.
* **image\_token** (`str`, *optional*, defaults to `"[IMG]"`) —
  Special token used to denote image location.
* **image\_break\_token** (`str`, *optional*, defaults to `"[IMG_BREAK]"`) —
  Special token used to denote the end of a line of pixels in an image.
* **image\_end\_token** (`str`, *optional*, defaults to `"[IMG_END]"`) —
  Special token used to denote the end of an image input.

Constructs a Pixtral processor which wraps a Pixtral image processor and a Pixtral tokenizer into a single processor.

[PixtralProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralProcessor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/pixtral.md)
