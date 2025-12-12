# Tokenizer

A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most
of the tokenizers are available in two flavors: a full python implementation and a "Fast" implementation based on the
Rust library [🤗 Tokenizers](https://github.com/huggingface/tokenizers). The "Fast" implementations allows:

1. a significant speed-up in particular when doing batched tokenization and
2. additional methods to map between the original string (character and words) and the token space (e.g. getting the
   index of the token comprising a given character or the span of characters corresponding to a given token).

The base classes [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) and [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)
implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and
"Fast" tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library
(downloaded from HuggingFace's AWS S3 repository). They both rely on
[PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) that contains the common methods.

[PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) and [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) thus implement the main
methods for using all the tokenizers:

- Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and
  encoding/decoding (i.e., tokenizing and converting to integers).
- Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece...).
- Managing special tokens (like mask, beginning-of-sentence, etc.): adding them, assigning them to attributes in the
  tokenizer for easy access and making sure they are not split during tokenization.

[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) holds the output of the
[PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)'s encoding methods (`__call__`,
`encode_plus` and `batch_encode_plus`) and is derived from a Python dictionary. When the tokenizer is a pure python
tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by
these methods (`input_ids`, `attention_mask`...). When the tokenizer is a "Fast" tokenizer (i.e., backed by
HuggingFace [tokenizers library](https://github.com/huggingface/tokenizers)), this class provides in addition
several advanced alignment methods which can be used to map between the original string (character and words) and the
token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding
to a given token).

## Multimodal Tokenizer

Apart from that each tokenizer can be a "multimodal" tokenizer which means that the tokenizer will hold all relevant special tokens
as part of tokenizer attributes for easier access. For example, if the tokenizer is loaded from a vision-language model like LLaVA, you will
be able to access `tokenizer.image_token_id` to obtain the special image token used as a placeholder.

To enable extra special tokens for any type of tokenizer, you have to add the following lines and save the tokenizer. Extra special tokens do not
have to be modality related and can be anything that the model often needs access to. In the below code, tokenizer at `output_dir` will have direct access
to three more special tokens.  

```python
vision_tokenizer = AutoTokenizer.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    extra_special_tokens={"image_token": "", "boi_token": "", "eoi_token": ""}
)
print(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
("", 32000)
```

## PreTrainedTokenizer[[transformers.PythonBackend]]

#### transformers.PythonBackend[[transformers.PythonBackend]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L400)

Base class for all slow tokenizers.

Inherits from [PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

Class attributes (overridden by derived classes)

- **vocab_files_names** (`dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
- **pretrained_vocab_files_map** (`dict[str, dict[str, str]]`) -- A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
- **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
- **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
- **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

__call__transformers.PythonBackend.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2469[{"name": "text", "val": ": Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None"}, {"name": "text_pair", "val": ": Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None"}, {"name": "text_target", "val": ": Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None"}, {"name": "text_pair_target", "val": ": Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": Union[bool, str, PaddingStrategy] = False"}, {"name": "truncation", "val": ": Union[bool, str, TruncationStrategy, None] = None"}, {"name": "max_length", "val": ": Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "is_split_into_words", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": Optional[int] = None"}, {"name": "padding_side", "val": ": Optional[str] = None"}, {"name": "return_tensors", "val": ": Optional[Union[str, TensorType]] = None"}, {"name": "return_token_type_ids", "val": ": Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "tokenizer_kwargs", "val": ": Optional[dict[str, Any]] = None"}, {"name": "**kwargs", "val": ""}]- **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_pair_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **tokenizer_kwargs** (`dict[str, Any]`, *optional*) --
  Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
  other kwargs, with explicit parameters taking precedence.

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **is_split_into_words** (`bool`, *optional*, defaults to `False`) --
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
- **padding_side** (`str`, *optional*) --
  The side on which the model should have padding applied. Should be selected between ['right', 'left'].
  Default value is picked from the class attribute of the same name.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.

- **return_token_type_ids** (`bool`, *optional*) --
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
- **return_overflowing_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
- **return_special_tokens_mask** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return special tokens mask information.
- **return_offsets_mapping** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
  Python's tokenizer, this method will raise `NotImplementedError`.
- **return_length**  (`bool`, *optional*, defaults to `False`) --
  Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, *optional*, defaults to `True`) --
  Whether or not to print more information and warnings.
- ****kwargs** -- passed to the `self.tokenize()` method0[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

**Parameters:**

model_max_length (`int`, *optional*) : The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with [from_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*) : The side on which the model should have truncation applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

chat_template (`str`, *optional*) : A Jinja template string that will be used to format lists of chat messages. See https://huggingface.co/docs/transformers/chat_templating for a full description.

model_input_names (`list[string]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

bos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the beginning of a sentence.

eos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the end of a sentence.

unk_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing an out-of-vocabulary token.

sep_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token separating two different sentences in the same input (used by BERT for instance).

pad_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

cls_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the class of the input (used by BERT for instance).

mask_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.

extra_special_tokens (list of `str` or `tokenizers.AddedToken`, *optional*) : A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end of the vocabulary.

split_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the special tokens should be split during the tokenization process. Passing will affect the internal state of the tokenizer. The default behavior is to not split special tokens. This means that if `` is the `bos_token`, then `tokenizer.tokenize("") = ['`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("")` will be give `['']`.

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)
#### add_tokens[[transformers.PythonBackend.add_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1210)

#TODO remove this from here! PreTrainedTOkeniuzerBase should be agnostic of AddedToken.

Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and
tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

Examples:

```python
# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))
```

**Parameters:**

new_tokens (`str`, `tokenizers.AddedToken` or a sequence of *str* or `tokenizers.AddedToken`) : Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string token to let you personalize its behavior: whether this token should only match against a single word, whether this token should strip all potential whitespaces on the left side, whether this token should strip all potential whitespaces on the right side, etc.

special_tokens (`bool`, *optional*, defaults to `False`) : Specifies if the token is special. This mostly changes the normalization behavior See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

**Returns:**

``int``

Number of tokens added to the vocabulary.
#### add_special_tokens[[transformers.PythonBackend.add_special_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1105)

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.

In order to do that, please use the [resize_token_embeddings()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) method.

Using `add_special_tokens` will ensure your special tokens can be used in several ways:

- Special tokens can be skipped when decoding using `skip_special_tokens = True`.
- Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
- You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
  makes it easy to develop model-agnostic training and fine-tuning scripts.

When possible, special tokens are already registered for provided pretrained models (for instance
[BertTokenizer](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer) `cls_token` is already registered to be `'[CLS]'` and XLM's one is also registered to be
`''`).

Examples:

```python
# Let's see how to add a new classification token to GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2")

special_tokens_dict = {"cls_token": ""}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))

assert tokenizer.cls_token == ""
```

**Parameters:**

special_tokens_dict (dictionary *str* to *str*, `tokenizers.AddedToken`, or `Sequence[Union[str, AddedToken]]`) : Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `extra_special_tokens`].  Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the `unk_token` to them).

replace_extra_special_tokens (`bool`, *optional*, defaults to `True`) : If `True`, the existing list of extra special tokens will be replaced by the list provided in `special_tokens_dict`. Otherwise, `extra_special_tokens` will be extended. In the former case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous `extra_special_tokens` are still added tokens, and will not be split by the model.

**Returns:**

``int``

Number of tokens added to the vocabulary.
#### apply_chat_template[[transformers.PythonBackend.apply_chat_template]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3026)

Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
determine the format and control tokens to use when converting.

**Parameters:**

conversation (Union[list[dict[str, str]], list[list[dict[str, str]]]]) : A list of dicts with "role" and "content" keys, representing the chat history so far.

tools (`list[Union[Dict, Callable]]`, *optional*) : A list of tools (callable functions) that will be accessible to the model. If the template does not support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema, giving the name, description and argument types for the tool. See our [tool use guide](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools) for more information.

documents (`list[dict[str, str]]`, *optional*) : A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation). If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing "title" and "text" keys.

chat_template (`str`, *optional*) : A Jinja template to use for this conversion. It is usually not necessary to pass anything to this argument, as the model's template will be used by default.

add_generation_prompt (bool, *optional*) : If this is set, a prompt with the token(s) that indicate the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model. Note that this argument will be passed to the chat template, and so it must be supported in the template for this argument to have any effect.

continue_final_message (bool, *optional*) : If this is set, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens. The model will continue this message rather than starting a new one. This allows you to "prefill" part of the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

tokenize (`bool`, defaults to `True`) : Whether to tokenize the output. If `False`, the output will be a string.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, defaults to `False`) : Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.

max_length (`int`, *optional*) : Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If not specified, the tokenizer's `max_length` attribute will be used as a default.

return_tensors (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

return_dict (`bool`, defaults to `False`) : Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.

tokenizer_kwargs (`dict[str : Any]`, *optional*): Additional kwargs to pass to the tokenizer.

return_assistant_tokens_mask (`bool`, defaults to `False`) : Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant, the mask will contain 1. For user and system tokens, the mask will contain 0. This functionality is only available for chat templates that support it via the `{% generation %}` keyword.

- ****kwargs** : Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

**Returns:**

``Union[list[int], Dict]``

A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
set, will return a dict of tokenizer outputs instead.
#### batch_decode[[transformers.PythonBackend.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2938)

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The `decode` method now handles batched input natively,
so you can use `decode` directly instead of `batch_decode`.

**Parameters:**

sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``list[str]``

The list of decoded sentences.
#### decode[[transformers.PythonBackend.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2890)

Converts a sequence of ids into a string, or a list of sequences into a list of strings,
using the tokenizer and vocabulary with options to remove special tokens and clean up
tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

**Parameters:**

token_ids (`Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]`) : A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``Union[str, list[str]]``

The decoded string for a single sequence, or a list of decoded strings for a
batch of sequences.
#### encode[[transformers.PythonBackend.encode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2283)

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

**Parameters:**

text (`str`, `list[str]` or `list[int]`) : The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method).

text_pair (`str`, `list[str]` or `list[int]`, *optional*) : Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method). 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided. - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

is_split_into_words (`bool`, *optional*, defaults to `False`) : Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return Numpy `np.ndarray` objects. 

- ****kwargs** : Passed along to the `.tokenize()` method.

**Returns:**

``list[int]`, `torch.Tensor`, or `np.ndarray``

The tokenized ids of the text.
#### push_to_hub[[transformers.PythonBackend.push_to_hub]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711)

Upload the tokenizer files to the 🤗 Model Hub.

Examples:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Push the tokenizer to your namespace with the name "my-finetuned-bert".
tokenizer.push_to_hub("my-finetuned-bert")

# Push the tokenizer to an organization with the name "my-finetuned-bert".
tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your tokenizer to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload tokenizer"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
#### build_inputs_with_special_tokens[[transformers.PythonBackend.build_inputs_with_special_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L858)

Build model inputs from a sequence or a pair of sequences by adding special tokens.

This method dynamically builds inputs based on the tokenizer's `special_tokens_pattern`:
- `"none"`: No special tokens
- `"cls_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
- `"eos"`: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
- `"bos"`: [BOS] seq0 or [BOS] seq0 [BOS] seq1
- `"bos_eos"`: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
- `"cls_double_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
- `"prefix_suffix"`: ` seq0 [seq1] ` (custom prefix/suffix stored on the tokenizer)

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs to which the special tokens will be added.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of input IDs with the appropriate special tokens.
#### create_token_type_ids_from_sequences[[transformers.PythonBackend.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1274)

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

This method dynamically builds the token type IDs based on the tokenizer's configuration attributes:
- `token_type_ids_pattern`: Pattern to use ("all_zeros" or "bert_style")
- `token_type_ids_include_special_tokens`: Whether to account for special tokens in length calculation

Examples:
```python
# All zeros pattern (default, used by RoBERTa, BART, etc.)
tokenizer.token_type_ids_pattern = "all_zeros"
# Returns: [0, 0, 0, ...] for both sequences

# BERT-style pattern (first sequence gets 0s, second gets 1s)
tokenizer.token_type_ids_pattern = "bert_style"
# Returns: [0, 0, 0, ..., 1, 1, 1, ...] for sequence pairs
```

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

Token type IDs according to the configured pattern.
#### get_added_vocab[[transformers.PythonBackend.get_added_vocab]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L489)

Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
something we should change.

**Returns:**

``dict[str, int]``

The added tokens.
#### get_special_tokens_mask[[transformers.PythonBackend.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L931)

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

This method dynamically builds the special tokens mask based on the tokenizer's `special_tokens_pattern`:
- `"none"`: No special tokens (default, returns all 0s)
- `"cls_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
- `"eos"`: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
- `"bos"`: [BOS] seq0 or [BOS] seq0 [BOS] seq1
- `"bos_eos"`: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
- `"cls_double_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
- `"prefix_suffix"`: ` seq0 [seq1] `

**Parameters:**

token_ids_0 (`list[int]`) : List of ids of the first sequence.

token_ids_1 (`list[int]`, *optional*) : List of ids of the second sequence.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### num_special_tokens_to_add[[transformers.PythonBackend.num_special_tokens_to_add]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L600)

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.

**Parameters:**

pair (`bool`, *optional*, defaults to `False`) : Whether the number of added tokens should be computed in the case of a sequence pair or a single sequence.

**Returns:**

``int``

Number of special tokens added to sequences.
#### prepare_for_model[[transformers.PythonBackend.prepare_for_model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1097)

Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.

**Parameters:**

ids : Tokenized input ids of the first sequence.

pair_ids : Tokenized input ids of the second sequence (optional).
#### prepare_for_tokenization[[transformers.PythonBackend.prepare_for_tokenization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L834)

Performs any necessary transformations before tokenization.

This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
`kwargs` at the end of the encoding process to be sure all the arguments have been used.

**Parameters:**

text (`str`) : The text to prepare.

is_split_into_words (`bool`, *optional*, defaults to `False`) : Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.

kwargs (`dict[str, Any]`, *optional*) : Keyword arguments to use for the tokenization.

**Returns:**

``tuple[str, dict[str, Any]]``

The prepared text and the unused kwargs.
#### save_vocabulary[[transformers.PythonBackend.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1338)

Default implementation for common vocabulary saving patterns.
Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges.
Returns empty tuple if no vocabulary exists.

Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models,
multiple vocabulary files, or special file formats).

**Parameters:**

save_directory (`str`) : The directory in which to save the vocabulary.

filename_prefix (`str`, *optional*) : An optional prefix to add to the named of the saved files.

**Returns:**

``tuple[str, ...]``

Paths to the files saved, or empty tuple if no files saved.
#### tokenize[[transformers.PythonBackend.tokenize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L623)

Converts a string into a sequence of tokens, using the tokenizer.

**Parameters:**

text : The sequence to be encoded.

- ****kwargs** : Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

**Returns:**

The list of tokens.
#### truncate_sequences[[transformers.PythonBackend.truncate_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1207)

Truncates sequences according to the specified strategy.

## PreTrainedTokenizerFast[[transformers.TokenizersBackend]]

The [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) depend on the [tokenizers](https://huggingface.co/docs/tokenizers) library. The tokenizers obtained from the 🤗 tokenizers library can be
loaded very simply into 🤗 transformers. Take a look at the [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) page to understand how this is done.

#### transformers.TokenizersBackend[[transformers.TokenizersBackend]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L80)

Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

Inherits from [PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

Handles all the shared methods for tokenization and special tokens, as well as methods for
downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

Class attributes (overridden by derived classes)

- **vocab_files_names** (`dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
- **pretrained_vocab_files_map** (`dict[str, dict[str, str]]`) -- A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
- **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
- **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
- **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

__call__transformers.TokenizersBackend.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2469[{"name": "text", "val": ": Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None"}, {"name": "text_pair", "val": ": Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None"}, {"name": "text_target", "val": ": Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None"}, {"name": "text_pair_target", "val": ": Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": Union[bool, str, PaddingStrategy] = False"}, {"name": "truncation", "val": ": Union[bool, str, TruncationStrategy, None] = None"}, {"name": "max_length", "val": ": Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "is_split_into_words", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": Optional[int] = None"}, {"name": "padding_side", "val": ": Optional[str] = None"}, {"name": "return_tensors", "val": ": Optional[Union[str, TensorType]] = None"}, {"name": "return_token_type_ids", "val": ": Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "tokenizer_kwargs", "val": ": Optional[dict[str, Any]] = None"}, {"name": "**kwargs", "val": ""}]- **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_pair_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **tokenizer_kwargs** (`dict[str, Any]`, *optional*) --
  Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
  other kwargs, with explicit parameters taking precedence.

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **is_split_into_words** (`bool`, *optional*, defaults to `False`) --
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
- **padding_side** (`str`, *optional*) --
  The side on which the model should have padding applied. Should be selected between ['right', 'left'].
  Default value is picked from the class attribute of the same name.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.

- **return_token_type_ids** (`bool`, *optional*) --
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
- **return_overflowing_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
- **return_special_tokens_mask** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return special tokens mask information.
- **return_offsets_mapping** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
  Python's tokenizer, this method will raise `NotImplementedError`.
- **return_length**  (`bool`, *optional*, defaults to `False`) --
  Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, *optional*, defaults to `True`) --
  Whether or not to print more information and warnings.
- ****kwargs** -- passed to the `self.tokenize()` method0[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

**Parameters:**

model_max_length (`int`, *optional*) : The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with [from_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*) : The side on which the model should have truncation applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

chat_template (`str`, *optional*) : A Jinja template string that will be used to format lists of chat messages. See https://huggingface.co/docs/transformers/chat_templating for a full description.

model_input_names (`list[string]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

bos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the beginning of a sentence.

eos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the end of a sentence.

unk_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing an out-of-vocabulary token.

sep_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token separating two different sentences in the same input (used by BERT for instance).

pad_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

cls_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the class of the input (used by BERT for instance).

mask_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.

extra_special_tokens (list of `str` or `tokenizers.AddedToken`, *optional*) : A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end of the vocabulary.

split_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the special tokens should be split during the tokenization process. Passing will affect the internal state of the tokenizer. The default behavior is to not split special tokens. This means that if `` is the `bos_token`, then `tokenizer.tokenize("") = ['`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("")` will be give `['']`. 

tokenizer_object ([tokenizers.Tokenizer](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer#tokenizers.Tokenizer)) : A [tokenizers.Tokenizer](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer#tokenizers.Tokenizer) object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) for more information.

tokenizer_file (`str`) : A path to a local JSON file representing a previously serialized [tokenizers.Tokenizer](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer#tokenizers.Tokenizer) object from 🤗 tokenizers.

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)
#### add_tokens[[transformers.TokenizersBackend.add_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1210)

#TODO remove this from here! PreTrainedTOkeniuzerBase should be agnostic of AddedToken.

Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and
tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

Examples:

```python
# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))
```

**Parameters:**

new_tokens (`str`, `tokenizers.AddedToken` or a sequence of *str* or `tokenizers.AddedToken`) : Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string token to let you personalize its behavior: whether this token should only match against a single word, whether this token should strip all potential whitespaces on the left side, whether this token should strip all potential whitespaces on the right side, etc.

special_tokens (`bool`, *optional*, defaults to `False`) : Specifies if the token is special. This mostly changes the normalization behavior See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

**Returns:**

``int``

Number of tokens added to the vocabulary.
#### add_special_tokens[[transformers.TokenizersBackend.add_special_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1105)

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.

In order to do that, please use the [resize_token_embeddings()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) method.

Using `add_special_tokens` will ensure your special tokens can be used in several ways:

- Special tokens can be skipped when decoding using `skip_special_tokens = True`.
- Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
- You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
  makes it easy to develop model-agnostic training and fine-tuning scripts.

When possible, special tokens are already registered for provided pretrained models (for instance
[BertTokenizer](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer) `cls_token` is already registered to be `'[CLS]'` and XLM's one is also registered to be
`''`).

Examples:

```python
# Let's see how to add a new classification token to GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2")

special_tokens_dict = {"cls_token": ""}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))

assert tokenizer.cls_token == ""
```

**Parameters:**

special_tokens_dict (dictionary *str* to *str*, `tokenizers.AddedToken`, or `Sequence[Union[str, AddedToken]]`) : Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `extra_special_tokens`].  Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the `unk_token` to them).

replace_extra_special_tokens (`bool`, *optional*, defaults to `True`) : If `True`, the existing list of extra special tokens will be replaced by the list provided in `special_tokens_dict`. Otherwise, `extra_special_tokens` will be extended. In the former case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous `extra_special_tokens` are still added tokens, and will not be split by the model.

**Returns:**

``int``

Number of tokens added to the vocabulary.
#### apply_chat_template[[transformers.TokenizersBackend.apply_chat_template]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3026)

Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
determine the format and control tokens to use when converting.

**Parameters:**

conversation (Union[list[dict[str, str]], list[list[dict[str, str]]]]) : A list of dicts with "role" and "content" keys, representing the chat history so far.

tools (`list[Union[Dict, Callable]]`, *optional*) : A list of tools (callable functions) that will be accessible to the model. If the template does not support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema, giving the name, description and argument types for the tool. See our [tool use guide](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools) for more information.

documents (`list[dict[str, str]]`, *optional*) : A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation). If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing "title" and "text" keys.

chat_template (`str`, *optional*) : A Jinja template to use for this conversion. It is usually not necessary to pass anything to this argument, as the model's template will be used by default.

add_generation_prompt (bool, *optional*) : If this is set, a prompt with the token(s) that indicate the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model. Note that this argument will be passed to the chat template, and so it must be supported in the template for this argument to have any effect.

continue_final_message (bool, *optional*) : If this is set, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens. The model will continue this message rather than starting a new one. This allows you to "prefill" part of the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

tokenize (`bool`, defaults to `True`) : Whether to tokenize the output. If `False`, the output will be a string.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, defaults to `False`) : Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.

max_length (`int`, *optional*) : Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If not specified, the tokenizer's `max_length` attribute will be used as a default.

return_tensors (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

return_dict (`bool`, defaults to `False`) : Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.

tokenizer_kwargs (`dict[str : Any]`, *optional*): Additional kwargs to pass to the tokenizer.

return_assistant_tokens_mask (`bool`, defaults to `False`) : Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant, the mask will contain 1. For user and system tokens, the mask will contain 0. This functionality is only available for chat templates that support it via the `{% generation %}` keyword.

- ****kwargs** : Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

**Returns:**

``Union[list[int], Dict]``

A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
set, will return a dict of tokenizer outputs instead.
#### batch_decode[[transformers.TokenizersBackend.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2938)

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The `decode` method now handles batched input natively,
so you can use `decode` directly instead of `batch_decode`.

**Parameters:**

sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``list[str]``

The list of decoded sentences.
#### decode[[transformers.TokenizersBackend.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2890)

Converts a sequence of ids into a string, or a list of sequences into a list of strings,
using the tokenizer and vocabulary with options to remove special tokens and clean up
tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

**Parameters:**

token_ids (`Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]`) : A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``Union[str, list[str]]``

The decoded string for a single sequence, or a list of decoded strings for a
batch of sequences.
#### encode[[transformers.TokenizersBackend.encode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2283)

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

**Parameters:**

text (`str`, `list[str]` or `list[int]`) : The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method).

text_pair (`str`, `list[str]` or `list[int]`, *optional*) : Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids` method). 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided. - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

is_split_into_words (`bool`, *optional*, defaults to `False`) : Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return Numpy `np.ndarray` objects. 

- ****kwargs** : Passed along to the `.tokenize()` method.

**Returns:**

``list[int]`, `torch.Tensor`, or `np.ndarray``

The tokenized ids of the text.
#### push_to_hub[[transformers.TokenizersBackend.push_to_hub]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711)

Upload the tokenizer files to the 🤗 Model Hub.

Examples:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Push the tokenizer to your namespace with the name "my-finetuned-bert".
tokenizer.push_to_hub("my-finetuned-bert")

# Push the tokenizer to an organization with the name "my-finetuned-bert".
tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your tokenizer to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload tokenizer"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
#### convert_to_native_format[[transformers.TokenizersBackend.convert_to_native_format]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L97)

s
Build a `tokenizers.Tokenizer` backend from the available serialization files (tokenizer.json, sentencepiece
models, tekken.json, vocab/merges).
#### get_added_vocab[[transformers.TokenizersBackend.get_added_vocab]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L508)

Returns the added tokens in the vocabulary as a dictionary of token to index.

**Returns:**

``dict[str, int]``

The added tokens.
#### num_special_tokens_to_add[[transformers.TokenizersBackend.num_special_tokens_to_add]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L605)

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.

**Parameters:**

pair (`bool`, *optional*, defaults to `False`) : Whether the number of added tokens should be computed in the case of a sequence pair or a single sequence.

**Returns:**

``int``

Number of special tokens added to sequences.
#### set_truncation_and_padding[[transformers.TokenizersBackend.set_truncation_and_padding]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L655)

Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
library) and restore the tokenizer settings afterwards.

The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
section.

**Parameters:**

padding_strategy ([PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy)) : The kind of padding that will be applied to the input

truncation_strategy ([TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy)) : The kind of truncation that will be applied to the input

max_length (`int`) : The maximum size of a sequence.

stride (`int`) : The stride to use when handling overflow.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.
#### train_new_from_iterator[[transformers.TokenizersBackend.train_new_from_iterator]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L922)

Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
as the current one.

**Parameters:**

text_iterator (generator of `list[str]`) : The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts if you have everything in memory.

vocab_size (`int`) : The size of the vocabulary you want for your tokenizer.

length (`int`, *optional*) : The total number of sequences in the iterator. This is used to provide meaningful progress tracking

new_special_tokens (list of `str` or `AddedToken`, *optional*) : A list of new special tokens to add to the tokenizer you are training.

special_tokens_map (`dict[str, str]`, *optional*) : If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special token name to new special token name in this argument.

kwargs (`dict[str, Any]`, *optional*) : Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

**Returns:**

`[PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)`

A new tokenizer of the same type as the original one, trained on
`text_iterator`.
#### update_post_processor[[transformers.TokenizersBackend.update_post_processor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L397)

Updates the underlying post processor with the current `bos_token` and `eos_token`.

## PythonBackend[[transformers.PythonBackend]]

#### transformers.PythonBackend[[transformers.PythonBackend]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L400)

Base class for all slow tokenizers.

Inherits from [PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

Class attributes (overridden by derived classes)

- **vocab_files_names** (`dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
- **pretrained_vocab_files_map** (`dict[str, dict[str, str]]`) -- A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
- **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
- **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
- **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

build_inputs_with_special_tokenstransformers.PythonBackend.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L858[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": list[int] | None = None"}]- **token_ids_0** (`list[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`list[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`list[int]`List of input IDs with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequences by adding special tokens.

This method dynamically builds inputs based on the tokenizer's `special_tokens_pattern`:
- `"none"`: No special tokens
- `"cls_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
- `"eos"`: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
- `"bos"`: [BOS] seq0 or [BOS] seq0 [BOS] seq1
- `"bos_eos"`: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
- `"cls_double_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
- `"prefix_suffix"`: ` seq0 [seq1] ` (custom prefix/suffix stored on the tokenizer)

**Parameters:**

model_max_length (`int`, *optional*) : The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with [from_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*) : The side on which the model should have truncation applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

chat_template (`str`, *optional*) : A Jinja template string that will be used to format lists of chat messages. See https://huggingface.co/docs/transformers/chat_templating for a full description.

model_input_names (`list[string]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

bos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the beginning of a sentence.

eos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the end of a sentence.

unk_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing an out-of-vocabulary token.

sep_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token separating two different sentences in the same input (used by BERT for instance).

pad_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

cls_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the class of the input (used by BERT for instance).

mask_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.

extra_special_tokens (list of `str` or `tokenizers.AddedToken`, *optional*) : A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end of the vocabulary.

split_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the special tokens should be split during the tokenization process. Passing will affect the internal state of the tokenizer. The default behavior is to not split special tokens. This means that if `` is the `bos_token`, then `tokenizer.tokenize("") = ['`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("")` will be give `['']`.

**Returns:**

``list[int]``

List of input IDs with the appropriate special tokens.
#### create_token_type_ids_from_sequences[[transformers.PythonBackend.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1274)

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

This method dynamically builds the token type IDs based on the tokenizer's configuration attributes:
- `token_type_ids_pattern`: Pattern to use ("all_zeros" or "bert_style")
- `token_type_ids_include_special_tokens`: Whether to account for special tokens in length calculation

Examples:
```python
# All zeros pattern (default, used by RoBERTa, BART, etc.)
tokenizer.token_type_ids_pattern = "all_zeros"
# Returns: [0, 0, 0, ...] for both sequences

# BERT-style pattern (first sequence gets 0s, second gets 1s)
tokenizer.token_type_ids_pattern = "bert_style"
# Returns: [0, 0, 0, ..., 1, 1, 1, ...] for sequence pairs
```

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

Token type IDs according to the configured pattern.
#### get_added_vocab[[transformers.PythonBackend.get_added_vocab]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L489)

Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
something we should change.

**Returns:**

``dict[str, int]``

The added tokens.
#### get_special_tokens_mask[[transformers.PythonBackend.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L931)

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

This method dynamically builds the special tokens mask based on the tokenizer's `special_tokens_pattern`:
- `"none"`: No special tokens (default, returns all 0s)
- `"cls_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
- `"eos"`: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
- `"bos"`: [BOS] seq0 or [BOS] seq0 [BOS] seq1
- `"bos_eos"`: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
- `"cls_double_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
- `"prefix_suffix"`: ` seq0 [seq1] `

**Parameters:**

token_ids_0 (`list[int]`) : List of ids of the first sequence.

token_ids_1 (`list[int]`, *optional*) : List of ids of the second sequence.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### num_special_tokens_to_add[[transformers.PythonBackend.num_special_tokens_to_add]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L600)

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.

**Parameters:**

pair (`bool`, *optional*, defaults to `False`) : Whether the number of added tokens should be computed in the case of a sequence pair or a single sequence.

**Returns:**

``int``

Number of special tokens added to sequences.
#### prepare_for_model[[transformers.PythonBackend.prepare_for_model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1097)

Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.

**Parameters:**

ids : Tokenized input ids of the first sequence.

pair_ids : Tokenized input ids of the second sequence (optional).
#### prepare_for_tokenization[[transformers.PythonBackend.prepare_for_tokenization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L834)

Performs any necessary transformations before tokenization.

This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
`kwargs` at the end of the encoding process to be sure all the arguments have been used.

**Parameters:**

text (`str`) : The text to prepare.

is_split_into_words (`bool`, *optional*, defaults to `False`) : Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.

kwargs (`dict[str, Any]`, *optional*) : Keyword arguments to use for the tokenization.

**Returns:**

``tuple[str, dict[str, Any]]``

The prepared text and the unused kwargs.
#### save_vocabulary[[transformers.PythonBackend.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1338)

Default implementation for common vocabulary saving patterns.
Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges.
Returns empty tuple if no vocabulary exists.

Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models,
multiple vocabulary files, or special file formats).

**Parameters:**

save_directory (`str`) : The directory in which to save the vocabulary.

filename_prefix (`str`, *optional*) : An optional prefix to add to the named of the saved files.

**Returns:**

``tuple[str, ...]``

Paths to the files saved, or empty tuple if no files saved.
#### tokenize[[transformers.PythonBackend.tokenize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L623)

Converts a string into a sequence of tokens, using the tokenizer.

**Parameters:**

text : The sequence to be encoded.

- ****kwargs** : Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

**Returns:**

The list of tokens.
#### truncate_sequences[[transformers.PythonBackend.truncate_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_python.py#L1207)

Truncates sequences according to the specified strategy.

## TokenizersBackend[[transformers.TokenizersBackend]]

#### transformers.TokenizersBackend[[transformers.TokenizersBackend]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L80)

Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

Inherits from [PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase).

Handles all the shared methods for tokenization and special tokens, as well as methods for
downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

Class attributes (overridden by derived classes)

- **vocab_files_names** (`dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
- **pretrained_vocab_files_map** (`dict[str, dict[str, str]]`) -- A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
- **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
- **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
- **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

convert_to_native_formattransformers.TokenizersBackend.convert_to_native_formathttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L97[{"name": "trust_remote_code", "val": " = False"}, {"name": "**kwargs", "val": ""}]
s
Build a `tokenizers.Tokenizer` backend from the available serialization files (tokenizer.json, sentencepiece
models, tekken.json, vocab/merges).

**Parameters:**

model_max_length (`int`, *optional*) : The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with [from_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*) : The side on which the model should have truncation applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

chat_template (`str`, *optional*) : A Jinja template string that will be used to format lists of chat messages. See https://huggingface.co/docs/transformers/chat_templating for a full description.

model_input_names (`list[string]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

bos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the beginning of a sentence.

eos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the end of a sentence.

unk_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing an out-of-vocabulary token.

sep_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token separating two different sentences in the same input (used by BERT for instance).

pad_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

cls_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the class of the input (used by BERT for instance).

mask_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.

extra_special_tokens (list of `str` or `tokenizers.AddedToken`, *optional*) : A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end of the vocabulary.

split_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the special tokens should be split during the tokenization process. Passing will affect the internal state of the tokenizer. The default behavior is to not split special tokens. This means that if `` is the `bos_token`, then `tokenizer.tokenize("") = ['`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("")` will be give `['']`. 

tokenizer_object ([tokenizers.Tokenizer](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer#tokenizers.Tokenizer)) : A [tokenizers.Tokenizer](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer#tokenizers.Tokenizer) object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) for more information.

tokenizer_file (`str`) : A path to a local JSON file representing a previously serialized [tokenizers.Tokenizer](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer#tokenizers.Tokenizer) object from 🤗 tokenizers.
#### get_added_vocab[[transformers.TokenizersBackend.get_added_vocab]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L508)

Returns the added tokens in the vocabulary as a dictionary of token to index.

**Returns:**

``dict[str, int]``

The added tokens.
#### num_special_tokens_to_add[[transformers.TokenizersBackend.num_special_tokens_to_add]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L605)

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.

**Parameters:**

pair (`bool`, *optional*, defaults to `False`) : Whether the number of added tokens should be computed in the case of a sequence pair or a single sequence.

**Returns:**

``int``

Number of special tokens added to sequences.
#### set_truncation_and_padding[[transformers.TokenizersBackend.set_truncation_and_padding]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L655)

Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
library) and restore the tokenizer settings afterwards.

The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
section.

**Parameters:**

padding_strategy ([PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy)) : The kind of padding that will be applied to the input

truncation_strategy ([TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy)) : The kind of truncation that will be applied to the input

max_length (`int`) : The maximum size of a sequence.

stride (`int`) : The stride to use when handling overflow.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.
#### train_new_from_iterator[[transformers.TokenizersBackend.train_new_from_iterator]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L922)

Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
as the current one.

**Parameters:**

text_iterator (generator of `list[str]`) : The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts if you have everything in memory.

vocab_size (`int`) : The size of the vocabulary you want for your tokenizer.

length (`int`, *optional*) : The total number of sequences in the iterator. This is used to provide meaningful progress tracking

new_special_tokens (list of `str` or `AddedToken`, *optional*) : A list of new special tokens to add to the tokenizer you are training.

special_tokens_map (`dict[str, str]`, *optional*) : If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special token name to new special token name in this argument.

kwargs (`dict[str, Any]`, *optional*) : Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

**Returns:**

`[PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)`

A new tokenizer of the same type as the original one, trained on
`text_iterator`.
#### update_post_processor[[transformers.TokenizersBackend.update_post_processor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L397)

Updates the underlying post processor with the current `bos_token` and `eos_token`.

## SentencePieceBackend[[transformers.SentencePieceBackend]]

#### transformers.SentencePieceBackend[[transformers.SentencePieceBackend]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_sentencepiece.py#L46)

Base class for SentencePiece-based tokenizers that load from sentencepiece.model files.

Inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

Class attributes (overridden by derived classes)

- **vocab_files_names** (`dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
- **pretrained_vocab_files_map** (`dict[str, dict[str, str]]`) -- A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
- **model_input_names** (`list[str]`) -- A list of inputs expected in the forward pass of the model.
- **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
- **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

convert_tokens_to_stringtransformers.SentencePieceBackend.convert_tokens_to_stringhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_sentencepiece.py#L233[{"name": "tokens", "val": ": list"}]
Converts a sequence of tokens (string) in a single string.

**Parameters:**

model_max_length (`int`, *optional*) : The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with [from_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*) : The side on which the model should have truncation applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

chat_template (`str`, *optional*) : A Jinja template string that will be used to format lists of chat messages. See https://huggingface.co/docs/transformers/chat_templating for a full description.

model_input_names (`list[string]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

bos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the beginning of a sentence.

eos_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the end of a sentence.

unk_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing an out-of-vocabulary token.

sep_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token separating two different sentences in the same input (used by BERT for instance).

pad_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

cls_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing the class of the input (used by BERT for instance).

mask_token (`str` or `tokenizers.AddedToken`, *optional*) : A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.

extra_special_tokens (list of `str` or `tokenizers.AddedToken`, *optional*) : A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end of the vocabulary.

split_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the special tokens should be split during the tokenization process. Passing will affect the internal state of the tokenizer. The default behavior is to not split special tokens. This means that if `` is the `bos_token`, then `tokenizer.tokenize("") = ['`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("")` will be give `['']`.
#### get_vocab[[transformers.SentencePieceBackend.get_vocab]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_sentencepiece.py#L105)

Returns vocab as a dict
#### save_vocabulary[[transformers.SentencePieceBackend.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_sentencepiece.py#L238)

Save the sentencepiece vocabulary (copy original file) to a directory.

**Parameters:**

save_directory (`str`) : The directory in which to save the vocabulary.

filename_prefix (`str`, *optional*) : An optional prefix to add to the named of the saved files.

**Returns:**

``tuple(str)``

Paths to the files saved.

## BatchEncoding[[transformers.BatchEncoding]]

#### transformers.BatchEncoding[[transformers.BatchEncoding]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L192)

Holds the output of the [__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__),
`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus` and
`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus` methods (tokens, attention_masks, etc).

This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
utility methods to map from word/character space to token space.

char_to_tokentransformers.BatchEncoding.char_to_tokenhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L539[{"name": "batch_or_char_index", "val": ": int"}, {"name": "char_index", "val": ": Optional[int] = None"}, {"name": "sequence_index", "val": ": int = 0"}]- **batch_or_char_index** (`int`) --
  Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
  the word in the sequence
- **char_index** (`int`, *optional*) --
  If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
  sequence.
- **sequence_index** (`int`, *optional*, defaults to 0) --
  If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
  or 1) the provided character index belongs to.0`int`Index of the token, or None if the char index refers to a whitespace only token and whitespace is
trimmed with `trim_offsets=True`.

Get the index of the token in the encoded output comprising a character in the original string for a sequence
of the batch.

Can be called as:

- `self.char_to_token(char_index)` if batch size is 1
- `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.

**Parameters:**

data (`dict`, *optional*) : Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods ('input_ids', 'attention_mask', etc.).

encoding (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, *optional*) : If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this information.

tensor_type (`Union[None, str, TensorType]`, *optional*) : You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at initialization.

prepend_batch_axis (`bool`, *optional*, defaults to `False`) : Whether or not to add a batch axis when converting to tensors (see `tensor_type` above). Note that this parameter has an effect if the parameter `tensor_type` is set, *otherwise has no effect*.

n_sequences (`Optional[int]`, *optional*) : You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at initialization.

**Returns:**

``int``

Index of the token, or None if the char index refers to a whitespace only token and whitespace is
trimmed with `trim_offsets=True`.
#### char_to_word[[transformers.BatchEncoding.char_to_word]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L626)

Get the word in the original string corresponding to a character in the original string of a sequence of the
batch.

Can be called as:

- `self.char_to_word(char_index)` if batch size is 1
- `self.char_to_word(batch_index, char_index)` if batch size is greater than 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.

**Parameters:**

batch_or_char_index (`int`) : Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the character in the original string.

char_index (`int`, *optional*) : If a batch index is provided in *batch_or_token_index*, this can be the index of the character in the original string.

sequence_index (`int`, *optional*, defaults to 0) : If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided character index belongs to.

**Returns:**

``int` or `list[int]``

Index or indices of the associated encoded token(s).
#### convert_to_tensors[[transformers.BatchEncoding.convert_to_tensors]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L665)

Convert the inner content to tensors.

**Parameters:**

tensor_type (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) : The type of tensors to use. If `str`, should be one of the values of the enum [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType). If `None`, no modification is done.

prepend_batch_axis (`int`, *optional*, defaults to `False`) : Whether or not to add the batch dimension during the conversion.
#### sequence_ids[[transformers.BatchEncoding.sequence_ids]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L327)

Return a list mapping the tokens to the id of their original sentences:

- `None` for special tokens added around or between sequences,
- `0` for tokens corresponding to words in the first sequence,
- `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
  encoded.

**Parameters:**

batch_index (`int`, *optional*, defaults to 0) : The index to access in the batch.

**Returns:**

``list[Optional[int]]``

A list indicating the sequence id corresponding to each token. Special tokens added
by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding
sequence.
#### to[[transformers.BatchEncoding.to]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L761)

Send all values to device by calling `v.to(device, non_blocking=non_blocking)` (PyTorch only).

**Parameters:**

device (`str` or `torch.device`) : The device to put the tensors on.

non_blocking (`bool`) : Whether to perform the copy asynchronously.

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

The same instance after modification.
#### token_to_chars[[transformers.BatchEncoding.token_to_chars]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L500)

Get the character span corresponding to an encoded token in a sequence of the batch.

Character spans are returned as a [CharSpan](/docs/transformers/main/en/internal/tokenization_utils#transformers.CharSpan) with:

- **start** -- Index of the first character in the original string associated to the token.
- **end** -- Index of the character following the last character in the original string associated to the
  token.

Can be called as:

- `self.token_to_chars(token_index)` if batch size is 1
- `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

**Parameters:**

batch_or_token_index (`int`) : Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the token in the sequence.

token_index (`int`, *optional*) : If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in the sequence.

**Returns:**

`[CharSpan](/docs/transformers/main/en/internal/tokenization_utils#transformers.CharSpan)`

Span of characters in the original string, or None, if the token
(e.g. , ) doesn't correspond to any chars in the origin string.
#### token_to_sequence[[transformers.BatchEncoding.token_to_sequence]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L370)

Get the index of the sequence represented by the given token. In the general use case, this method returns `0`
for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

Can be called as:

- `self.token_to_sequence(token_index)` if batch size is 1
- `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
words are defined by the user). In this case it allows to easily associate encoded tokens with provided
tokenized words.

**Parameters:**

batch_or_token_index (`int`) : Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of the token in the sequence.

token_index (`int`, *optional*) : If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the sequence.

**Returns:**

``int``

Index of the word in the input sequence.
#### token_to_word[[transformers.BatchEncoding.token_to_word]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L409)

Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

Can be called as:

- `self.token_to_word(token_index)` if batch size is 1
- `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
words are defined by the user). In this case it allows to easily associate encoded tokens with provided
tokenized words.

**Parameters:**

batch_or_token_index (`int`) : Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the token in the sequence.

token_index (`int`, *optional*) : If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the sequence.

**Returns:**

``int``

Index of the word in the input sequence.
#### tokens[[transformers.BatchEncoding.tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L309)

Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
integer indices) at a given batch index (only works for the output of a fast tokenizer).

**Parameters:**

batch_index (`int`, *optional*, defaults to 0) : The index to access in the batch.

**Returns:**

``list[str]``

The list of tokens at that index.
#### word_ids[[transformers.BatchEncoding.word_ids]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L351)

Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

**Parameters:**

batch_index (`int`, *optional*, defaults to 0) : The index to access in the batch.

**Returns:**

``list[Optional[int]]``

A list indicating the word corresponding to each token. Special tokens added by the
tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
(several tokens will be mapped to the same word index if they are parts of that word).
#### word_to_chars[[transformers.BatchEncoding.word_to_chars]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L581)

Get the character span in the original string corresponding to given word in a sequence of the batch.

Character spans are returned as a CharSpan NamedTuple with:

- start: index of the first character in the original string
- end: index of the character following the last character in the original string

Can be called as:

- `self.word_to_chars(word_index)` if batch size is 1
- `self.word_to_chars(batch_index, word_index)` if batch size is greater or equal to 1

**Parameters:**

batch_or_word_index (`int`) : Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of the word in the sequence

word_index (`int`, *optional*) : If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the sequence.

sequence_index (`int`, *optional*, defaults to 0) : If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided word index belongs to.

**Returns:**

``CharSpan` or `list[CharSpan]``

Span(s) of the associated character or characters in the string. CharSpan
are NamedTuple with:

- start: index of the first character associated to the token in the original string
- end: index of the character following the last character associated to the token in the original
  string
#### word_to_tokens[[transformers.BatchEncoding.word_to_tokens]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L447)

Get the encoded token span corresponding to a word in a sequence of the batch.

Token spans are returned as a [TokenSpan](/docs/transformers/main/en/internal/tokenization_utils#transformers.TokenSpan) with:

- **start** -- Index of the first token.
- **end** -- Index of the token following the last token.

Can be called as:

- `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
- `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
  1

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.

**Parameters:**

batch_or_word_index (`int`) : Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of the word in the sequence.

word_index (`int`, *optional*) : If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the sequence.

sequence_index (`int`, *optional*, defaults to 0) : If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0 or 1) the provided word index belongs to.

**Returns:**

`([TokenSpan](/docs/transformers/main/en/internal/tokenization_utils#transformers.TokenSpan), *optional*)`

Span of tokens in the encoded sequence. Returns
`None` if no tokens correspond to the word. This can happen especially when the token is a special token
that has been used to format the tokenization. For example when we add a class token at the very beginning
of the tokenization.
