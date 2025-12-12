# Generation

Each framework has a generate method for text generation implemented in their respective `GenerationMixin` class:

- PyTorch [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) is implemented in [GenerationMixin](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin).

You can parameterize the generate method with a [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig) class instance. Please refer to this class for the complete list of generation parameters, which control the behavior of the generation method.

To learn how to inspect a model's generation configuration, what are the defaults, how to change the parameters ad hoc,
and how to create and save a customized generation configuration, refer to the
[text generation strategies guide](../generation_strategies). The guide also explains how to use related features,
like token streaming.

## GenerationConfig[[transformers.GenerationConfig]]

#### transformers.GenerationConfig[[transformers.GenerationConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L82)

Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

- *greedy decoding* if `num_beams=1` and `do_sample=False`
- *multinomial sampling* if `num_beams=1` and `do_sample=True`
- *beam-search decoding* if `num_beams>1` and `do_sample=False`
- *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
- *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
description of the possible manipulations, as well as examples of their usage.

from_pretrainedtransformers.GenerationConfig.from_pretrainedhttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L771[{"name": "pretrained_model_name", "val": ": str | os.PathLike"}, {"name": "config_file_name", "val": ": str | os.PathLike | None = None"}, {"name": "cache_dir", "val": ": str | os.PathLike | None = None"}, {"name": "force_download", "val": ": bool = False"}, {"name": "local_files_only", "val": ": bool = False"}, {"name": "token", "val": ": str | bool | None = None"}, {"name": "revision", "val": ": str = 'main'"}, {"name": "**kwargs", "val": ""}]- **pretrained_model_name** (`str` or `os.PathLike`) --
  This can be either:

  - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
    huggingface.co.
  - a path to a *directory* containing a configuration file saved using the
    [save_pretrained()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.save_pretrained) method, e.g., `./my_model_directory/`.
- **config_file_name** (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`) --
  Name of the generation configuration JSON file to be loaded from `pretrained_model_name`.
- **cache_dir** (`str` or `os.PathLike`, *optional*) --
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
- **force_download** (`bool`, *optional*, defaults to `False`) --
  Whether or not to force to (re-)download the configuration files and override the cached versions if
  they exist.
- **proxies** (`dict[str, str]`, *optional*) --
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
  'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
- **token** (`str` or `bool`, *optional*) --
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
- **revision** (`str`, *optional*, defaults to `"main"`) --
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

  

  To test a pull request you made on the Hub, you can pass `revision="refs/pr/"`.

  

- **return_unused_kwargs** (`bool`, *optional*, defaults to `False`) --
  If `False`, then this function returns just the final configuration object.

  If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
  dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
  part of `kwargs` which has not been used to update `config` and is otherwise ignored.
- **subfolder** (`str`, *optional*, defaults to `""`) --
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.
- **kwargs** (`dict[str, Any]`, *optional*) --
  The values in kwargs of any keys which are configuration attributes will be used to override the loaded
  values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
  by the `return_unused_kwargs` keyword parameter.0[GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)The configuration object instantiated from this pretrained model.

Instantiate a [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig) from a generation configuration file.

Examples:

```python
>>> from transformers import GenerationConfig

>>> # Download configuration from huggingface.co and cache.
>>> generation_config = GenerationConfig.from_pretrained("openai-community/gpt2")

>>> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
>>> generation_config.save_pretrained("./test/saved_model/")
>>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

>>> # You can also specify configuration names to your generation configuration file
>>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
>>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

>>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
>>> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
>>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
...     "openai-community/gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
... )
>>> generation_config.top_k
1

>>> unused_kwargs
{'foo': False}
```

**Returns:**

`[GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)`

The configuration object instantiated from this pretrained model.
#### from_model_config[[transformers.GenerationConfig.from_model_config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L1103)

Instantiates a [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig) from a [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig). This function is useful to convert legacy
[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) objects, which may contain generation parameters, into a stand-alone [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig).

**Parameters:**

model_config (`PreTrainedConfig | dict`) : The model config that will be used to instantiate the generation config.

**Returns:**

`[GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)`

The configuration object instantiated from those parameters.
#### save_pretrained[[transformers.GenerationConfig.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L712)

Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the
[from_pretrained()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.from_pretrained) class method.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the configuration JSON file will be saved (will be created if it does not exist).

config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`) : Name of the generation configuration JSON file to be saved in `save_directory`.

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### update[[transformers.GenerationConfig.update]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L1148)

Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
returning all the unused kwargs.

**Parameters:**

kwargs (`dict[str, Any]`) : Dictionary of attributes to tentatively update this class.

**Returns:**

``dict[str, Any]``

Dictionary containing all the key-value pairs that were not used to update the instance.
#### validate[[transformers.GenerationConfig.validate]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L540)

Validates the values of the attributes of the [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig) instance. Raises exceptions in the presence
of parameterization that can be detected as incorrect from the configuration instance alone.

Note that some parameters not validated here are best validated at generate runtime, as they may depend on
other inputs and/or the model, such as parameters related to the generation length.

**Parameters:**

strict (bool) : If True, raise an exception for any issues found. If False, only log issues.
#### get_generation_mode[[transformers.GenerationConfig.get_generation_mode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L475)

Returns the generation mode triggered by the [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig) instance.

**Parameters:**

assistant_model (`PreTrainedModel`, *optional*) : The assistant model to be used for assisted generation. If set, the generation mode will be assisted generation.

**Returns:**

``GenerationMode``

The generation mode triggered by the instance.

## GenerationMixin[[transformers.GenerationMixin]]

#### transformers.GenerationMixin[[transformers.GenerationMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L340)

A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes.
Inheriting from this class causes the model to have special generation-related behavior, such as loading a
`GenerationConfig` at initialization time or ensuring `generate`-related tests are run in `transformers` CI.

A model class should inherit from `GenerationMixin` to enable calling methods like `generate`, or when it
has defined a custom `generate` method that relies on `GenerationMixin`, directly or indirectly, which
approximately shares the same interface to public methods like `generate`. Three examples:
- `LlamaForCausalLM` should inherit from `GenerationMixin` to enable calling `generate` and other public
  methods in the mixin;
- `BlipForQuestionAnswering` has a custom `generate` method that approximately shares the same interface as
  `GenerationMixin.generate` (it has a few extra arguments, and the same output). That function also calls
  `GenerationMixin.generate` indirectly, through an inner model. As such, `BlipForQuestionAnswering` should
  inherit from `GenerationMixin` to benefit from all generation-related automation in our codebase;
- `BarkModel` has a custom `generate` method and one of its inner models calls `GenerationMixin.generate`.
  However, its `generate` does not share the same interface as `GenerationMixin.generate`. In this case,
  `BarkModel` should NOT inherit from `GenerationMixin`, as it breaks the `generate` interface.

The class exposes [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate), which can be used for:
- *greedy decoding* if `num_beams=1` and `do_sample=False`
- *multinomial sampling* if `num_beams=1` and `do_sample=True`
- *beam-search decoding* if `num_beams>1` and `do_sample=False`
- *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
- *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

generatetransformers.GenerationMixin.generatehttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L2284[{"name": "inputs", "val": ": torch.Tensor | None = None"}, {"name": "generation_config", "val": ": transformers.generation.configuration_utils.GenerationConfig | None = None"}, {"name": "logits_processor", "val": ": transformers.generation.logits_process.LogitsProcessorList | None = None"}, {"name": "stopping_criteria", "val": ": transformers.generation.stopping_criteria.StoppingCriteriaList | None = None"}, {"name": "prefix_allowed_tokens_fn", "val": ": collections.abc.Callable[[int, torch.Tensor], list[int]] | None = None"}, {"name": "synced_gpus", "val": ": bool | None = None"}, {"name": "assistant_model", "val": ": typing.Optional[ForwardRef('PreTrainedModel')] = None"}, {"name": "streamer", "val": ": typing.Optional[ForwardRef('BaseStreamer')] = None"}, {"name": "negative_prompt_ids", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "use_model_defaults", "val": ": bool | None = None"}, {"name": "custom_generate", "val": ": str | collections.abc.Callable | None = None"}, {"name": "**kwargs", "val": ""}]- **inputs** (`torch.Tensor` of varying shape depending on the modality, *optional*) --
  The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
  method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
  should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
  `input_ids`, `input_values`, `input_features`, or `pixel_values`.
- **generation_config** ([GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) --
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which has the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)'s
  default values, whose documentation should be checked to parameterize generation.
- **logits_processor** (`LogitsProcessorList`, *optional*) --
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
- **stopping_criteria** (`StoppingCriteriaList`, *optional*) --
  Custom stopping criteria that complements the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
  sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
  intended for advanced users.
- **prefix_allowed_tokens_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) --
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
  `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
  on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
  for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
- **synced_gpus** (`bool`, *optional*) --
  Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
  to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
  deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
- **assistant_model** (`PreTrainedModel`, *optional*) --
  An assistant model that can be used to accelerate generation. The assistant model must have the exact
  same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
  is much faster than running generation with the model you're calling generate from. As such, the
  assistant model should be much smaller.
- **streamer** (`BaseStreamer`, *optional*) --
  Streamer object that will be used to stream the generated sequences. Generated tokens are passed
  through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
- **negative_prompt_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  The negative prompt needed for some processors such as CFG. The batch size must match the input batch
  size. This is an experimental feature, subject to breaking API changes in future versions.
- **negative_prompt_attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Attention_mask for `negative_prompt_ids`.
- **use_model_defaults** (`bool`, *optional*) --
  When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
  generation configuration (`model.generation_config`), as opposed to the global defaults
  (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
  `True`.
- **custom_generate** (`str` or `Callable`, *optional*) --
  One of the following:
  - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
    `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
    repository fully replaces the generation logic, and the return type may differ.
  - `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
  - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
    run the decoding loop.
  For more information, see [the docs](../../generation_strategies#custom-generation-methods).
- **kwargs** (`dict[str, Any]`, *optional*) --
  Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
  specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.0[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`A [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) types are:

- [GenerateDecoderOnlyOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput),
- [GenerateBeamDecoderOnlyOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput)

If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) types are:

- [GenerateEncoderDecoderOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
- [GenerateBeamEncoderDecoderOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model's default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](../generation_strategies).

**Parameters:**

inputs (`torch.Tensor` of varying shape depending on the modality, *optional*) : The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs` should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of `input_ids`, `input_values`, `input_features`, or `pixel_values`.

generation_config ([GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) : The generation configuration to be used as base parametrization for the generation call. `**kwargs` passed to generate matching the attributes of `generation_config` will override them. If `generation_config` is not provided, the default will be used, which has the following loading priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)'s default values, whose documentation should be checked to parameterize generation.

logits_processor (`LogitsProcessorList`, *optional*) : Custom logits processors that complement the default logits processors built from arguments and generation config. If a logit processor is passed that is already created with the arguments or a generation config an error is thrown. This feature is intended for advanced users.

stopping_criteria (`StoppingCriteriaList`, *optional*) : Custom stopping criteria that complements the default stopping criteria built from arguments and a generation config. If a stopping criteria is passed that is already created with the arguments or a generation config an error is thrown. If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is intended for advanced users.

prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*) : If provided, this function constraints the beam search to allowed tokens only at each step. If not provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful for constrained generation conditioned on the prefix, as described in [Autoregressive Entity Retrieval](https://huggingface.co/papers/2010.00904).

synced_gpus (`bool`, *optional*) : Whether to continue running the while loop until max_length. Unless overridden, this flag will be set to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.

assistant_model (`PreTrainedModel`, *optional*) : An assistant model that can be used to accelerate generation. The assistant model must have the exact same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model is much faster than running generation with the model you're calling generate from. As such, the assistant model should be much smaller.

streamer (`BaseStreamer`, *optional*) : Streamer object that will be used to stream the generated sequences. Generated tokens are passed through `streamer.put(token_ids)` and the streamer is responsible for any further processing.

negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : The negative prompt needed for some processors such as CFG. The batch size must match the input batch size. This is an experimental feature, subject to breaking API changes in future versions.

negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Attention_mask for `negative_prompt_ids`.

use_model_defaults (`bool`, *optional*) : When it is `True`, unset parameters in `generation_config` will be set to the model-specific default generation configuration (`model.generation_config`), as opposed to the global defaults (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be `True`.

custom_generate (`str` or `Callable`, *optional*) : One of the following: - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at `custom_generate/generate.py` in that repository instead of the standard `generate` method. The repository fully replaces the generation logic, and the return type may differ. - `str` (local repository path): same as above but from a local path, `trust_remote_code` not required. - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to run the decoding loop. For more information, see [the docs](../../generation_strategies#custom-generation-methods).

kwargs (`dict[str, Any]`, *optional*) : Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

**Returns:**

`[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor``

A [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) types are:

- [GenerateDecoderOnlyOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput),
- [GenerateBeamDecoderOnlyOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput)

If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) types are:

- [GenerateEncoderDecoderOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
- [GenerateBeamEncoderDecoderOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)
#### compute_transition_scores[[transformers.GenerationMixin.compute_transition_scores]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1437)

Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
used). This is a convenient method to quickly obtain the scores of the selected tokens at generation time.

Examples:

```python
>>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
>>> import numpy as np

>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> tokenizer.pad_token_id = tokenizer.eos_token_id
>>> inputs = tokenizer(["Today is"], return_tensors="pt")

>>> # Example 1: Print the scores for each token generated with Greedy Search
>>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, normalize_logits=True
... )
>>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
>>> # encoder-decoder models, like BART or T5.
>>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
>>> generated_tokens = outputs.sequences[:, input_length:]
>>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
...     # | token | token string | log probability | probability
...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
|   262 |  the     | -1.414 | 24.33%
|  1110 |  day     | -2.609 | 7.36%
|   618 |  when    | -2.010 | 13.40%
|   356 |  we      | -1.859 | 15.58%
|   460 |  can     | -2.508 | 8.14%

>>> # Example 2: Reconstruct the sequence scores from Beam Search
>>> outputs = model.generate(
...     **inputs,
...     max_new_tokens=5,
...     num_beams=4,
...     num_return_sequences=4,
...     return_dict_in_generate=True,
...     output_scores=True,
... )
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
... )
>>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
>>> # Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
>>> # use case, you might want to recompute it with `normalize_logits=True`.
>>> # Tip 2: the output length does NOT include the input length
>>> output_length = np.sum(transition_scores.numpy() >> length_penalty = model.generation_config.length_penalty
>>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
>>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
True
```

**Parameters:**

sequences (`torch.LongTensor`) : The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.

scores (`tuple(torch.FloatTensor)`) : Transition scores for each vocabulary token at each generation step. Beam transition scores consisting of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.

beam_indices (`torch.LongTensor`, *optional*) : Beam indices of generated token id at each generation step. `torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at generate-time.

normalize_logits (`bool`, *optional*, defaults to `False`) : Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

**Returns:**

``torch.Tensor``

A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
the transition scores (logits)

## ContinuousMixin[[transformers.ContinuousMixin]]

#### transformers.ContinuousMixin[[transformers.ContinuousMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L1117)

Mixin class for models to add continuous batching capabilities.

generate_batchtransformers.ContinuousMixin.generate_batchhttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L1195[{"name": "inputs", "val": ": list"}, {"name": "generation_config", "val": ": transformers.generation.configuration_utils.GenerationConfig | None = None"}, {"name": "num_q_padding_intervals", "val": ": int = 0"}, {"name": "num_kv_padding_intervals", "val": ": int = 0"}, {"name": "allow_prefix_sharing", "val": ": bool = True"}, {"name": "record_timestamps", "val": ": bool = False"}, {"name": "progress_bar", "val": ": bool = True"}, {"name": "**kwargs", "val": ""}]- **inputs** -- List of input token sequences (prompts)
- **generation_config** -- Optional generation configuration
- **num_q_padding_intervals** -- Number of intervals used to pad the query dimension
- **num_kv_padding_intervals** -- Number of intervals used to pad the keys/values dimension
- **allow_prefix_sharing** -- A flag to allow prefix sharing if the model has only full attention layers
- **record_timestamps** -- If set to true, the requests will have a timestamp for each token generated
- **progress_bar** -- If set to true, a progress bar will be displayed
- ****kwargs** -- Additional generation parameters0`dict[str, GenerationOutput]`a dictionary of request ids to GenerationOutput objects
Generate sequences for a batch of prompts using continuous batching.

**Parameters:**

inputs : List of input token sequences (prompts)

generation_config : Optional generation configuration

num_q_padding_intervals : Number of intervals used to pad the query dimension

num_kv_padding_intervals : Number of intervals used to pad the keys/values dimension

allow_prefix_sharing : A flag to allow prefix sharing if the model has only full attention layers

record_timestamps : If set to true, the requests will have a timestamp for each token generated

progress_bar : If set to true, a progress bar will be displayed

- ****kwargs** : Additional generation parameters

**Returns:**

``dict[str, GenerationOutput]``

a dictionary of request ids to GenerationOutput objects
#### init_continuous_batching[[transformers.ContinuousMixin.init_continuous_batching]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L1150)

Initialize a manager for continuous batching inference.

**Parameters:**

generation_config : An optional generation configuration, which may contain a CompileConfig object

manual_eviction : Whether to manually evict requests from the cache

max_queue_size : Maximum size of the input request queue

num_q_padding_intervals : Number of intervals used to pad the query dimension

num_kv_padding_intervals : Number of intervals used to pad the keys/values dimension

allow_prefix_sharing : A flag to allow prefix sharing if the model has only full attention layers

**Returns:**

``ContinuousBatchingManager``

The manager instance to add requests and retrieve results.

## ContinuousBatchingManager[[transformers.ContinuousBatchingManager]]

#### transformers.ContinuousBatchingManager[[transformers.ContinuousBatchingManager]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L740)

Manager for handling continuous batching of generation requests.

This class provides the user interface for submitting generation requests,
retrieving results, and managing the background generation thread.

add_requesttransformers.ContinuousBatchingManager.add_requesthttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L906[{"name": "input_ids", "val": ": list"}, {"name": "request_id", "val": ": str | None = None"}, {"name": "max_new_tokens", "val": ": int | None = None"}, {"name": "streaming", "val": ": bool = False"}, {"name": "record_timestamps", "val": ": bool = False"}]- **input_ids** -- Input token IDs to use as prompt
- **request_id** -- Optional custom request ID (auto-generated if None)
- ****kwargs** -- Additional generation parameters0strThe request ID
Add a new generation request to the queue.

**Parameters:**

input_ids : Input token IDs to use as prompt

request_id : Optional custom request ID (auto-generated if None)

- ****kwargs** : Additional generation parameters

**Returns:**

`str`

The request ID
#### cancel_request[[transformers.ContinuousBatchingManager.cancel_request]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L958)

Cancel a request by its ID.

**Parameters:**

request_id : The ID of the request to cancel
#### evict_request_from_cache[[transformers.ContinuousBatchingManager.evict_request_from_cache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L1108)

Evict a request from the cache. It is assumed that the request is already finished.
#### get_result[[transformers.ContinuousBatchingManager.get_result]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L968)

Retrieve one result from the output queue.

**Parameters:**

timeout : Maximum time to wait for a result

**Returns:**

`Optional[GenerationOutput]`

The result data or None if timeout
#### is_running[[transformers.ContinuousBatchingManager.is_running]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L857)

Check if the background generation thread is running.
#### join[[transformers.ContinuousBatchingManager.join]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L891)

Wait for the background thread to finish.

**Parameters:**

timeout : Maximum time to wait for the thread to stop
#### request_id_iter[[transformers.ContinuousBatchingManager.request_id_iter]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L997)

Iterate over results matching a specific request id as they become available.
#### start[[transformers.ContinuousBatchingManager.start]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L847)

Start the background generation thread.
#### stop[[transformers.ContinuousBatchingManager.stop]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py#L862)

Signal the background thread to stop.

**Parameters:**

block : Whether to wait for the thread to stop

timeout : Maximum time to wait for the thread to stop

## Scheduler[[transformers.generation.Scheduler]]

#### transformers.generation.Scheduler[[transformers.generation.Scheduler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L24)

Abstract base class for scheduling requests in the continuous batch processor. Schedulers manage the lifecycle of
requests from when they are added to the waiting queue to when they are scheduled for processing. Different
schedulers implement different strategies for prioritizing and batching requests.

add_waiting_requesttransformers.generation.Scheduler.add_waiting_requesthttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L40[{"name": "state", "val": ": RequestState"}]
Adds a request to the waiting list.
#### clear_cancelled_requests[[transformers.generation.Scheduler.clear_cancelled_requests]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L87)

Remove all cancelled requests from active and waiting queues.
#### finish_request[[transformers.generation.Scheduler.finish_request]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L64)

Completes processing of a request and optionally frees its allocated cache blocks. This method is called
when a request has finished generation or encountered an error.
#### get_active_request_static_outputs[[transformers.generation.Scheduler.get_active_request_static_outputs]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L74)

Gets generated tokens for an active request.
#### has_pending_requests[[transformers.generation.Scheduler.has_pending_requests]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L59)

Checks if there are requests ready to be processed.
#### request_is_cancelled[[transformers.generation.Scheduler.request_is_cancelled]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L101)

Checks if a request has been cancelled or removed.
#### schedule_batch[[transformers.generation.Scheduler.schedule_batch]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L53)

Schedules requests for the next batch based on available token budget. This method selects which requests
should be processed in the current batch, considering the token budget and the scheduler's prioritization rules.
The token_budget is the maximum number of tokens that can be processed in this batch.
#### set_request_cancellation[[transformers.generation.Scheduler.set_request_cancellation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L81)

Marks a request for cancellation.

## FIFOScheduler[[transformers.generation.FIFOScheduler]]

#### transformers.generation.FIFOScheduler[[transformers.generation.FIFOScheduler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L178)

This scheduler processes requests in the order they arrive, meaning decoding requests has priority over
prefilling requests. Additionally, it includes a safety margin mechanism to prevent cache exhaustion. By default,
when 80% of the cache is full, new requests will not be scheduled to prioritize decoding active requests.

## PrefillFirstScheduler[[transformers.generation.PrefillFirstScheduler]]

#### transformers.generation.PrefillFirstScheduler[[transformers.generation.PrefillFirstScheduler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/scheduler.py#L257)

Scheduler that prioritizes split prefill requests over decoding requests. This scheduler ensures that split
prefill requests (which are continuations of partially processed prompts) are completed before processing new
decoding requests.
