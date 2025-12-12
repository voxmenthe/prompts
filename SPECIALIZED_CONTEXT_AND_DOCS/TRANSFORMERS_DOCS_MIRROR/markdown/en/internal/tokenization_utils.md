# Utilities for Tokenizers

This page lists all the utility functions used by the tokenizers, mainly the class
[PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) that implements the common methods between
[PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) and the mixin
[SpecialTokensMixin](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.SpecialTokensMixin).

Most of those are only useful if you are studying the code of the tokenizers in the library.

## PreTrainedTokenizerBase

### class transformers.PreTrainedTokenizerBase

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1375)

( \*\*kwargs  )

Parameters

* **model\_max\_length** (`int`, *optional*) —
  The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
  loaded with [from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the
  value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will
  default to VERY\_LARGE\_INTEGER (`int(1e30)`).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **truncation\_side** (`str`, *optional*) —
  The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **chat\_template** (`str`, *optional*) —
  A Jinja template string that will be used to format lists of chat messages. See
  <https://huggingface.co/docs/transformers/chat_templating> for a full description.
* **model\_input\_names** (`list[string]`, *optional*) —
  The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
  `"attention_mask"`). Default value is picked from the class attribute of the same name.
* **bos\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the beginning of a sentence. Will be associated to `self.bos_token` and
  `self.bos_token_id`.
* **eos\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the end of a sentence. Will be associated to `self.eos_token` and
  `self.eos_token_id`.
* **unk\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing an out-of-vocabulary token. Will be associated to `self.unk_token` and
  `self.unk_token_id`.
* **sep\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token separating two different sentences in the same input (used by BERT for instance). Will be
  associated to `self.sep_token` and `self.sep_token_id`.
* **pad\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
  attention mechanisms or loss computation. Will be associated to `self.pad_token` and `self.pad_token_id`.
* **cls\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the class of the input (used by BERT for instance). Will be associated to
  `self.cls_token` and `self.cls_token_id`.
* **mask\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing a masked token (used by masked-language modeling pretraining objectives, like
  BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.
* **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, *optional*) —
  A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
  `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end
  of the vocabulary.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should cleanup the spaces that were added when splitting the input text during the
  tokenization process.
* **split\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the special tokens should be split during the tokenization process. Passing will affect the
  internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
  `<s>` is the `bos_token`, then `tokenizer.tokenize("<s>") = ['<s>`]. Otherwise, if
  `split_special_tokens=True`, then `tokenizer.tokenize("<s>")` will be give `['<','s', '>']`.

Base class for [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast).

Handles shared (mostly boiler plate) methods for those two classes.

Class attributes (overridden by derived classes)

* **vocab\_files\_names** (`dict[str, str]`) — A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
* **pretrained\_vocab\_files\_map** (`dict[str, dict[str, str]]`) — A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
* **model\_input\_names** (`list[str]`) — A list of inputs expected in the forward pass of the model.
* **padding\_side** (`str`) — The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
* **truncation\_side** (`str`) — The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828)

( text: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
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
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
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
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
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

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

#### apply\_chat\_template

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1518)

( conversation: typing.Union[list[dict[str, str]], list[list[dict[str, str]]]] tools: typing.Optional[list[typing.Union[dict, typing.Callable]]] = None documents: typing.Optional[list[dict[str, str]]] = None chat\_template: typing.Optional[str] = None add\_generation\_prompt: bool = False continue\_final\_message: bool = False tokenize: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: bool = False max\_length: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_dict: bool = False return\_assistant\_tokens\_mask: bool = False tokenizer\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  ) → `Union[list[int], Dict]`

Parameters

* **conversation** (Union[list[dict[str, str]], list[list[dict[str, str]]]]) — A list of dicts
  with “role” and “content” keys, representing the chat history so far.
* **tools** (`list[Union[Dict, Callable]]`, *optional*) —
  A list of tools (callable functions) that will be accessible to the model. If the template does not
  support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
  giving the name, description and argument types for the tool. See our
  [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
  for more information.
* **documents** (`list[dict[str, str]]`, *optional*) —
  A list of dicts representing documents that will be accessible to the model if it is performing RAG
  (retrieval-augmented generation). If the template does not support RAG, this argument will have no
  effect. We recommend that each document should be a dict containing “title” and “text” keys. Please
  see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
  for examples of passing documents with chat templates.
* **chat\_template** (`str`, *optional*) —
  A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
  argument, as the model’s template will be used by default.
* **add\_generation\_prompt** (bool, *optional*) —
  If this is set, a prompt with the token(s) that indicate
  the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
  Note that this argument will be passed to the chat template, and so it must be supported in the
  template for this argument to have any effect.
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
  + `'tf'`: Return TensorFlow `tf.Tensor` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.
* **return\_dict** (`bool`, defaults to `False`) —
  Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
* **tokenizer\_kwargs** (`dict[str -- Any]`, *optional*): Additional kwargs to pass to the tokenizer.
* **return\_assistant\_tokens\_mask** (`bool`, defaults to `False`) —
  Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
  the mask will contain 1. For user and system tokens, the mask will contain 0.
  This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
* \***\*kwargs** — Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

Returns

`Union[list[int], Dict]`

A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
set, will return a dict of tokenizer outputs instead.

Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer’s chat\_template attribute to
determine the format and control tokens to use when converting.

#### as\_target\_tokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L4001)

( )

Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
sequence-to-sequence models that need a slightly different processing for the labels.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `list[str]`

Parameters

* **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`list[str]`

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### batch\_encode\_plus

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3144)

( batch\_text\_or\_text\_pairs: typing.Union[list[str], list[tuple[str, str]], list[list[str]], list[tuple[list[str], list[str]]], list[list[int]], list[tuple[list[int], list[int]]]] add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True split\_special\_tokens: bool = False \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **batch\_text\_or\_text\_pairs** (`list[str]`, `list[tuple[str, str]]`, `list[list[str]]`, `list[tuple[list[str], list[str]]]`, and for not-fast tokenizers, also `list[list[int]]`, `list[tuple[list[int], list[int]]]`) —
  Batch of sequences or pair of sequences to be encoded. This can be a list of
  string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
  details in `encode_plus`).
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
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
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
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
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
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
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

Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

This method is deprecated, `__call__` should be used instead.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3456)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) — The first tokenized sequence.
* **token\_ids\_1** (`list[int]`, *optional*) — The second tokenized sequence.

Returns

`list[int]`

The model input with special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens.

This implementation does not add special tokens and this method should be overridden in a subclass.

#### clean\_up\_tokenization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3944)

( out\_string: str  ) → `str`

Parameters

* **out\_string** (`str`) — The text to clean up.

Returns

`str`

The cleaned-up string.

Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3820)

( tokens: list  ) → `str`

Parameters

* **tokens** (`list[str]`) — The token to join in a string.

Returns

`str`

The joined tokens.

Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
often want to remove sub-word tokenization artifacts at the same time.

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

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3867)

( token\_ids: typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `str`

Parameters

* **token\_ids** (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`str`

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### encode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2667)

( text: typing.Union[str, list[str], list[int]] text\_pair: typing.Union[str, list[str], list[int], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  ) → `list[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`

Parameters

* **text** (`str`, `list[str]` or `list[int]`) —
  The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
  `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).
* **text\_pair** (`str`, `list[str]` or `list[int]`, *optional*) —
  Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
  the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).
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
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
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
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* \***\*kwargs** — Passed along to the `.tokenize()` method.

Returns

`list[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

#### encode\_message\_with\_chat\_template

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1697)

( message: dict conversation\_history: typing.Optional[list[dict[str, str]]] = None \*\*kwargs  ) → `list[int]`

Parameters

* **message** (`dict`) —
  A dictionary with “role” and “content” keys, representing the message to tokenize.
* **conversation\_history** (`list[dict]`, *optional*) —
  A list of dicts with “role” and “content” keys, representing the chat history so far. If you are
  tokenizing messages one by one, you should pass the previous messages in the conversation here.
* \***\*kwargs** —
  Additional kwargs to pass to the `apply_chat_template` method.

Returns

`list[int]`

A list of token ids representing the tokenized message.

Tokenize a single message. This method is a convenience wrapper around `apply_chat_template` that allows you
to tokenize messages one by one. This is useful for things like token-by-token streaming.
This method is not guaranteed to be perfect. For some models, it may be impossible to robustly tokenize
single messages. For example, if the chat template adds tokens after each message, but also has a prefix that
is added to the entire chat, it will be impossible to distinguish a chat-start-token from a message-start-token.
In these cases, this method will do its best to find the correct tokenization, but it may not be perfect.
**Note:** This method does not support `add_generation_prompt`. If you want to add a generation prompt,
you should do it separately after tokenizing the conversation.

#### encode\_plus

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3044)

( text: typing.Union[str, list[str], list[int]] text\_pair: typing.Union[str, list[str], list[int], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]` or (for non-fast tokenizers) `list[int]`) —
  The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
  `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).
* **text\_pair** (`str`, `list[str]` or `list[int]`, *optional*) —
  Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
  the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
  method).
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
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
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
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
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
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
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

Tokenize and prepare for the model a sequence or a pair of sequences.

This method is deprecated, `__call__` should be used instead.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1807)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] \*init\_inputs cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' trust\_remote\_code = False \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
    using the [save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  + (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
    file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
    `./my_model_directory/vocab.txt`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
  exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **token** (`str` or *bool*, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **local\_files\_only** (`bool`, *optional*, defaults to `False`) —
  Whether or not to only rely on local files and not to attempt to download any files.
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **subfolder** (`str`, *optional*) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
  facebook/rag-token-base), specify it here.
* **inputs** (additional positional arguments, *optional*) —
  Will be passed along to the Tokenizer `__init__` method.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the Tokenizer `__init__` method. Can be used to set special tokens like `bos_token`,
  `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
  `additional_special_tokens`. See parameters in the `__init__` for more details.

Instantiate a [PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) (or a derived class) from a predefined
tokenizer.

Passing `token=True` is required when you want to use a private model.

Examples:


```
# We can't instantiate directly the base class *PreTrainedTokenizerBase* so let's show our examples on a derived class: BertTokenizer
# Download vocabulary from huggingface.co and cache.
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Download vocabulary from huggingface.co (user-uploaded) and cache.
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

# If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
tokenizer = BertTokenizer.from_pretrained("./test/saved_model/")

# If the tokenizer uses a single vocabulary file, you can point directly to this file
tokenizer = BertTokenizer.from_pretrained("./test/saved_model/my_vocab.txt")

# You can link tokens to special vocabulary when instantiating
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", unk_token="<unk>")
# You should be sure '<unk>' is in the vocabulary when doing that.
# Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
assert tokenizer.unk_token == "<unk>"
```

#### get\_chat\_template

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1753)

( chat\_template: typing.Optional[str] = None tools: typing.Optional[list[dict]] = None  ) → `str`

Parameters

* **chat\_template** (`str`, *optional*) —
  A Jinja template or the name of a template to use for this conversion.
  It is usually not necessary to pass anything to this argument,
  as the model’s template will be used by default.
* **tools** (`list[Dict]`, *optional*) —
  A list of tools (callable functions) that will be accessible to the model. If the template does not
  support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
  giving the name, description and argument types for the tool. See our
  [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
  for more information.

Returns

`str`

The chat template string.

Retrieve the chat template string used for tokenizing chat messages. This template is used
internally by the `apply_chat_template` method and can also be used externally to retrieve the model’s chat
template for better generation tracking.

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

#### get\_vocab

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1506)

( ) → `dict[str, int]`

Returns

`dict[str, int]`

The vocabulary.

Returns the vocabulary as a dictionary of token to index.

`tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
vocab.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3253)

( encoded\_inputs: typing.Union[transformers.tokenization\_utils\_base.BatchEncoding, list[transformers.tokenization\_utils\_base.BatchEncoding], dict[str, list[int]], dict[str, list[list[int]]], list[dict[str, list[int]]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True max\_length: typing.Optional[int] = None pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_attention\_mask: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None verbose: bool = True  )

Parameters

* **encoded\_inputs** ([BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding), list of [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding), `dict[str, list[int]]`, `dict[str, list[list[int]]` or `list[dict[str, list[int]]]`) —
  Tokenized inputs. Can represent one input ([BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) or `dict[str, list[int]]`) or a batch of
  tokenized inputs (list of [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding), *dict[str, list[list[int]]]* or *list[dict[str,
  list[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
  collate function.

  Instead of `list[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
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
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.

Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
in the batch.

Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
`self.pad_token_id` and `self.pad_token_type_id`).

Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
text followed by a call to the `pad` method to get a padded encoding.

If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.

#### prepare\_for\_model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3476)

( ids: list pair\_ids: typing.Optional[list[int]] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True prepend\_batch\_axis: bool = False \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **ids** (`list[int]`) —
  Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
  `convert_tokens_to_ids` methods.
* **pair\_ids** (`list[int]`, *optional*) —
  Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
  and `convert_tokens_to_ids` methods.
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
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
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
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
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
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
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

Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair\_ids*
different than `None` and *truncation\_strategy = longest\_first* or `True`, it is not possible to return
overflowing tokens. Such a combination of arguments will raise an error.

#### prepare\_seq2seq\_batch

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L4040)

( src\_texts: list tgt\_texts: typing.Optional[list[str]] = None max\_length: typing.Optional[int] = None max\_target\_length: typing.Optional[int] = None padding: str = 'longest' return\_tensors: typing.Optional[str] = None truncation: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **src\_texts** (`list[str]`) —
  List of documents to summarize or source language texts.
* **tgt\_texts** (`list`, *optional*) —
  List of summaries or target language texts.
* **max\_length** (`int`, *optional*) —
  Controls the maximum length for encoder inputs (documents to summarize or source language texts) If
  left unset or set to `None`, this will use the predefined model maximum length if a maximum length is
  required by one of the truncation/padding parameters. If the model has no specific maximum input length
  (like XLNet) truncation/padding to a maximum length will be deactivated.
* **max\_target\_length** (`int`, *optional*) —
  Controls the maximum length of decoder inputs (target language texts or summaries) If left unset or set
  to `None`, this will use the max\_length value.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `True`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* \***\*kwargs** —
  Additional keyword arguments passed along to `self.__call__`.

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to the encoder.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model.
* **labels** — List of token ids for tgt\_texts.

The full set of keys `[input_ids, attention_mask, labels]`, will only be returned if tgt\_texts is passed.
Otherwise, input\_ids, attention\_mask will be the only keys.

Prepare model inputs for translation. For best performance, translate one sentence at a time.

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847)

( repo\_id: str use\_temp\_dir: typing.Optional[bool] = None commit\_message: typing.Optional[str] = None private: typing.Optional[bool] = None token: typing.Union[bool, str, NoneType] = None max\_shard\_size: typing.Union[str, int, NoneType] = '5GB' create\_pr: bool = False safe\_serialization: bool = True revision: typing.Optional[str] = None commit\_description: typing.Optional[str] = None tags: typing.Optional[list[str]] = None \*\*deprecated\_kwargs  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your tokenizer to. It should contain your organization name
  when pushing to a given organization.
* **use\_temp\_dir** (`bool`, *optional*) —
  Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
  Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
* **commit\_message** (`str`, *optional*) —
  Message to commit while pushing. Will default to `"Upload tokenizer"`.
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
  is not specified.
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"5GB"`) —
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
  Google Colab instances without any CPU OOM issues.
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.
* **safe\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to convert the model weights in safetensors format for safer serialization.
* **revision** (`str`, *optional*) —
  Branch to push the uploaded files to.
* **commit\_description** (`str`, *optional*) —
  The description of the commit that will be created
* **tags** (`list[str]`, *optional*) —
  List of tags to push on the Hub.

Upload the tokenizer files to the 🤗 Model Hub.

Examples:


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Push the tokenizer to your namespace with the name "my-finetuned-bert".
tokenizer.push_to_hub("my-finetuned-bert")

# Push the tokenizer to an organization with the name "my-finetuned-bert".
tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```

#### register\_for\_auto\_class

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L4018)

( auto\_class = 'AutoTokenizer'  )

Parameters

* **auto\_class** (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`) —
  The auto class to register this new tokenizer with.

Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
library are already mapped with `AutoTokenizer`.

#### save\_chat\_templates

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2370)

( save\_directory: typing.Union[str, os.PathLike] tokenizer\_config: dict filename\_prefix: typing.Optional[str] save\_jinja\_files: bool  )

Writes chat templates out to the save directory if we’re using the new format, and removes them from
the tokenizer config if present. If we’re using the legacy format, it doesn’t write any files, and instead
writes the templates to the tokenizer config in the correct format.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2425)

( save\_directory: typing.Union[str, os.PathLike] legacy\_format: typing.Optional[bool] = None filename\_prefix: typing.Optional[str] = None push\_to\_hub: bool = False \*\*kwargs  ) → A tuple of `str`

Parameters

* **save\_directory** (`str` or `os.PathLike`) — The path to a directory where the tokenizer will be saved.
* **legacy\_format** (`bool`, *optional*) —
  Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
  format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
  added\_tokens files.

  If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with
  “slow” tokenizers (not powered by the *tokenizers* library), so the tokenizer will not be able to be
  loaded in the corresponding “slow” tokenizer.

  If `True`, will save the tokenizer in legacy format. If the “slow” tokenizer doesn’t exits, a value
  error is raised.
* **filename\_prefix** (`str`, *optional*) —
  A prefix to add to the names of the files saved by the tokenizer.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Returns

A tuple of `str`

The files saved.

Save the full tokenizer state.

This method make sure the full tokenizer can then be re-loaded using the
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` class method..

Warning,None This won’t save modifications you may have applied to the tokenizer after the instantiation (for
instance, modifying `tokenizer.do_lower_case` after creation).

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2629)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  ) → `Tuple(str)`

Parameters

* **save\_directory** (`str`) —
  The directory in which to save the vocabulary.
* **filename\_prefix** (`str`, *optional*) —
  An optional prefix to add to the named of the saved files.

Returns

`Tuple(str)`

Paths to the files saved.

Save only the vocabulary of the tokenizer (vocabulary + added tokens).

This method won’t save the configuration and special token mappings of the tokenizer. Use
`_save_pretrained()` to save the whole state of the tokenizer.

#### tokenize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2647)

( text: str pair: typing.Optional[str] = None add\_special\_tokens: bool = False \*\*kwargs  ) → `list[str]`

Parameters

* **text** (`str`) —
  The sequence to be encoded.
* **pair** (`str`, *optional*) —
  A second sequence to be encoded with the first.
* **add\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add the special tokens associated with the corresponding model.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific encode method. See details in
  [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__)

Returns

`list[str]`

The list of tokens.

Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.

#### truncate\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3614)

( ids: list pair\_ids: typing.Optional[list[int]] = None num\_tokens\_to\_remove: int = 0 truncation\_strategy: typing.Union[str, transformers.tokenization\_utils\_base.TruncationStrategy] = 'longest\_first' stride: int = 0  ) → `tuple[list[int], list[int], list[int]]`

Parameters

* **ids** (`list[int]`) —
  Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
  `convert_tokens_to_ids` methods.
* **pair\_ids** (`list[int]`, *optional*) —
  Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
  and `convert_tokens_to_ids` methods.
* **num\_tokens\_to\_remove** (`int`, *optional*, defaults to 0) —
  Number of tokens to remove using the truncation strategy.
* **truncation\_strategy** (`str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `'longest_first'`) —
  The strategy to follow for truncation. Can be:
  + `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will truncate
    token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
    batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
    than the model maximum admissible input size).
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a positive number, the overflowing tokens returned will contain some tokens from the main
  sequence returned. The value of this argument defines the number of additional tokens.

Returns

`tuple[list[int], list[int], list[int]]`

The truncated `ids`, the truncated `pair_ids` and the list of
overflowing tokens. Note: The *longest\_first* strategy returns empty list of overflowing tokens if a pair
of sequences (or a batch of pairs) is provided.

Truncates a sequence pair in-place following the strategy.

## SpecialTokensMixin

### class transformers.SpecialTokensMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L818)

( verbose = False \*\*kwargs  )

Parameters

* **bos\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the beginning of a sentence.
* **eos\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the end of a sentence.
* **unk\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing an out-of-vocabulary token.
* **sep\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token separating two different sentences in the same input (used by BERT for instance).
* **pad\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
  attention mechanisms or loss computation.
* **cls\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the class of the input (used by BERT for instance).
* **mask\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing a masked token (used by masked-language modeling pretraining objectives, like
  BERT).
* **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, *optional*) —
  A tuple or a list of additional tokens, which will be marked as `special`, meaning that they will be
  skipped when decoding if `skip_special_tokens` is set to `True`.

A mixin derived by [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) to handle specific behaviors related to
special tokens. In particular, this class hold the attributes which can be used to directly access these special
tokens in a model-independent manner and allow to set and update the special tokens.

#### add\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L890)

( special\_tokens\_dict: dict replace\_additional\_special\_tokens = True  ) → `int`

Parameters

* **special\_tokens\_dict** (dictionary *str* to *str*, `tokenizers.AddedToken`, or `Sequence[Union[str, AddedToken]]`) —
  Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
  `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].

  Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
  assign the index of the `unk_token` to them).
* **replace\_additional\_special\_tokens** (`bool`, *optional*,, defaults to `True`) —
  If `True`, the existing list of additional special tokens will be replaced by the list provided in
  `special_tokens_dict`. Otherwise, `self._special_tokens_map["additional_special_tokens"]` is just extended. In the former
  case, the tokens will NOT be removed from the tokenizer’s full vocabulary - they are only being flagged
  as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
  `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
  `additional_special_tokens` are still added tokens, and will not be split by the model.

Returns

`int`

Number of tokens added to the vocabulary.

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.

In order to do that, please use the [resize\_token\_embeddings()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) method.

Using `add_special_tokens` will ensure your special tokens can be used in several ways:

* Special tokens can be skipped when decoding using `skip_special_tokens = True`.
* Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
* You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
  makes it easy to develop model-agnostic training and fine-tuning scripts.

When possible, special tokens are already registered for provided pretrained models (for instance
[BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) `cls_token` is already registered to be `'[CLS]'` and XLM’s one is also registered to be
`'</s>'`).

Examples:


```
# Let's see how to add a new classification token to GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2")

special_tokens_dict = {"cls_token": "<CLS>"}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))

assert tokenizer.cls_token == "<CLS>"
```

#### add\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L994)

( new\_tokens: typing.Union[str, tokenizers.AddedToken, collections.abc.Sequence[typing.Union[str, tokenizers.AddedToken]]] special\_tokens: bool = False  ) → `int`

Parameters

* **new\_tokens** (`str`, `tokenizers.AddedToken` or a sequence of *str* or `tokenizers.AddedToken`) —
  Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
  token to let you personalize its behavior: whether this token should only match against a single word,
  whether this token should strip all potential whitespaces on the left side, whether this token should
  strip all potential whitespaces on the right side, etc.
* **special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Can be used to specify if the token is a special token. This mostly change the normalization behavior
  (special tokens like CLS or [MASK] are usually not lower-cased for instance).

  See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

Returns

`int`

Number of tokens added to the vocabulary.

Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
it with indices starting from length of the current vocabulary and will be isolated before the tokenization
algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
not treated in the same way.

Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
of the model so that its embedding matrix matches the tokenizer.

In order to do that, please use the [resize\_token\_embeddings()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) method.

Examples:


```
# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))
```

#### sanitize\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L882)

( )

The `sanitize_special_tokens` is now deprecated kept for backward compatibility and will be removed in
transformers v5.

## Enums and namedtuples

### class transformers.tokenization\_utils\_base.TruncationStrategy

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L154)

( value names = None module = None qualname = None type = None start = 1  )

Possible values for the `truncation` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for tab-completion in
an IDE.

### class transformers.CharSpan

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L166)

( start: int end: int  )

Parameters

* **start** (`int`) — Index of the first character in the original string.
* **end** (`int`) — Index of the character following the last character in the original string.

Character span in the original string.

### class transformers.TokenSpan

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L179)

( start: int end: int  )

Parameters

* **start** (`int`) — Index of the first token in the span.
* **end** (`int`) — Index of the token following the last token in the span.

Token span in an encoded string (list of tokens).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/tokenization_utils.md)
