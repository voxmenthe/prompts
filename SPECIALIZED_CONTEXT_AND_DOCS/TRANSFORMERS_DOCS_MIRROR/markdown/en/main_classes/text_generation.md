# Generation

Each framework has a generate method for text generation implemented in their respective `GenerationMixin` class:

* PyTorch [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) is implemented in [GenerationMixin](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin).

You can parameterize the generate method with a [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) class instance. Please refer to this class for the complete list of generation parameters, which control the behavior of the generation method.

To learn how to inspect a model’s generation configuration, what are the defaults, how to change the parameters ad hoc,
and how to create and save a customized generation configuration, refer to the
[text generation strategies guide](../generation_strategies). The guide also explains how to use related features,
like token streaming.

## GenerationConfig

### class transformers.GenerationConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L82)

( \*\*kwargs  )

Parameters that control the length of the output

* **max\_length** (`int`, *optional*, defaults to 20) —
  The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
  `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
* **max\_new\_tokens** (`int`, *optional*) —
  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
* **min\_length** (`int`, *optional*, defaults to 0) —
  The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
  `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
* **min\_new\_tokens** (`int`, *optional*) —
  The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
* **early\_stopping** (`bool` or `str`, *optional*, defaults to `False`) —
  Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
  `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
  heuristic is applied and the generation stops when is it very unlikely to find better candidates;
  `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
  beam search algorithm).
* **max\_time** (`float`, *optional*) —
  The maximum amount of time you allow the computation to run for in seconds. generation will still finish
  the current pass after allocated time has been passed.
* **stop\_strings** (`str or list[str]`, *optional*) —
  A string or a list of strings that should terminate generation if the model outputs them.

Parameters that control the generation strategy used

* **do\_sample** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use sampling ; use greedy decoding otherwise.
* **num\_beams** (`int`, *optional*, defaults to 1) —
  Number of beams for beam search. 1 means no beam search.
* **num\_beam\_groups** (`int`, *optional*, defaults to 1) —
  Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
  [this paper](https://huggingface.co/papers/1610.02424) for more details.

Parameters that control the cache

* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should use the past last key/values attentions (if applicable to the model) to
  speed up decoding.
* **cache\_implementation** (`str`, *optional*, default to `None`) —
  Name of the cache class that will be instantiated in `generate`, for faster decoding. Possible values are:
  + `"dynamic"`: [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache)
  + `"static"`: [StaticCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.StaticCache)
  + `"offloaded"`: `DynamicCache(offloaded=True)`
  + `"offloaded_static"`: `StaticCache(offloaded=True)`
  + `"quantized"`: [QuantizedCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.QuantizedCache)

  If none is specified, we will use the default cache for the model (which is often [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache)). See
  our [cache documentation](https://huggingface.co/docs/transformers/en/kv_cache) for further information.
* **cache\_config** (`dict`, *optional*, default to `None`) —
  Arguments used in the key-value cache class can be passed in `cache_config`.
* **return\_legacy\_cache** (`bool`, *optional*, default to `True`) —
  Whether to return the legacy or new format of the cache when `DynamicCache` is used by default.

Parameters for manipulation of the model output logits

* **temperature** (`float`, *optional*, defaults to 1.0) —
  The value used to module the next token probabilities. This value is set in a model’s `generation_config.json` file. If it isn’t set, the default value is 1.0
* **top\_k** (`int`, *optional*, defaults to 50) —
  The number of highest probability vocabulary tokens to keep for top-k-filtering. This value is set in a model’s `generation_config.json` file. If it isn’t set, the default value is 50.
* **top\_p** (`float`, *optional*, defaults to 1.0) —
  If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
  `top_p` or higher are kept for generation. This value is set in a model’s `generation_config.json` file. If it isn’t set, the default value is 1.0
* **min\_p** (`float`, *optional*) —
  Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
  value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
  the 0.99-0.8 range (use the opposite of normal `top_p` values).
* **typical\_p** (`float`, *optional*, defaults to 1.0) —
  Local typicality measures how similar the conditional probability of predicting a target token next is to
  the expected conditional probability of predicting a random token next, given the partial text already
  generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
  add up to `typical_p` or higher are kept for generation. See [this
  paper](https://huggingface.co/papers/2202.00666) for more details.
* **epsilon\_cutoff** (`float`, *optional*, defaults to 0.0) —
  If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
  `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
  size of the model. See [Truncation Sampling as Language Model
  Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
* **eta\_cutoff** (`float`, *optional*, defaults to 0.0) —
  Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
  0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
  probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
  depending on the size of the model. See [Truncation Sampling as Language Model
  Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
* **diversity\_penalty** (`float`, *optional*, defaults to 0.0) —
  This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a
  particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
* **repetition\_penalty** (`float`, *optional*, defaults to 1.0) —
  The parameter for repetition penalty. 1.0 means no penalty. See [this
  paper](https://huggingface.co/papers/1909.05858) for more details.
* **encoder\_repetition\_penalty** (`float`, *optional*, defaults to 1.0) —
  The parameter for encoder\_repetition\_penalty. An exponential penalty on sequences that are not in the
  original input. 1.0 means no penalty.
* **length\_penalty** (`float`, *optional*, defaults to 1.0) —
  Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
  the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
  likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
  `length_penalty` < 0.0 encourages shorter sequences.
* **no\_repeat\_ngram\_size** (`int`, *optional*, defaults to 0) —
  If set to int > 0, all ngrams of that size can only occur once.
* **bad\_words\_ids** (`list[list[int]]`, *optional*) —
  List of list of token ids that are not allowed to be generated. Check
  [NoBadWordsLogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.NoBadWordsLogitsProcessor) for further documentation and examples.
* **force\_words\_ids** (`list[list[int]]` or `list[list[list[int]]]`, *optional*) —
  List of token ids that must be generated. If given a `list[list[int]]`, this is treated as a simple list of
  words that must be included, the opposite to `bad_words_ids`. If given `list[list[list[int]]]`, this
  triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
  can allow different forms of each word.
* **renormalize\_logits** (`bool`, *optional*, defaults to `False`) —
  Whether to renormalize the logits after applying all the logits processors (including the custom
  ones). It’s highly recommended to set this flag to `True` as the search algorithms suppose the score logits
  are normalized but some logit processors break the normalization.
* **constraints** (`list[Constraint]`, *optional*) —
  Custom constraints that can be added to the generation to ensure that the output will contain the use of
  certain tokens as defined by `Constraint` objects, in the most sensible way possible.
* **forced\_bos\_token\_id** (`int`, *optional*, defaults to `model.config.forced_bos_token_id`) —
  The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
  multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
  language token.
* **forced\_eos\_token\_id** (`int` or list[int]`, *optional*, defaults to` model.config.forced\_eos\_token\_id`) -- The id of the token to force as the last generated token when` max\_length` is reached. Optionally, use a
  list to set multiple *end-of-sequence* tokens.
* **remove\_invalid\_values** (`bool`, *optional*, defaults to `model.config.remove_invalid_values`) —
  Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
  Note that using `remove_invalid_values` can slow down generation.
* **exponential\_decay\_length\_penalty** (`tuple(int, float)`, *optional*) —
  This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
  generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
  penalty starts and `decay_factor` represents the factor of exponential decay
* **suppress\_tokens** (`list[int]`, *optional*) —
  A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their
  log probs to `-inf` so that they are not sampled.
* **begin\_suppress\_tokens** (`list[int]`, *optional*) —
  A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit
  processor will set their log probs to `-inf` so that they are not sampled.
* **sequence\_bias** (`dict[tuple[int], float]`, *optional*)) —
  Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
  sequence being selected, while negative biases do the opposite. Check
  [SequenceBiasLogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor) for further documentation and examples.
* **token\_healing** (`bool`, *optional*, defaults to `False`) —
  Heal tail tokens of prompts by replacing them with their appropriate extensions.
  This enhances the quality of completions for prompts affected by greedy tokenization bias.
* **guidance\_scale** (`float`, *optional*) —
  The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
  Higher guidance scale encourages the model to generate samples that are more closely linked to the input
  prompt, usually at the expense of poorer quality.
* **watermarking\_config** (`BaseWatermarkingConfig` or `dict`, *optional*) —
  Arguments used to watermark the model outputs by adding a small bias to randomly selected set of “green”
  tokens. See the docs of [SynthIDTextWatermarkingConfig](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.SynthIDTextWatermarkingConfig) and [WatermarkingConfig](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.WatermarkingConfig) for more
  details. If passed as `Dict`, it will be converted to a `WatermarkingConfig` internally.

Parameters that define the output variables of generate

* **num\_return\_sequences** (`int`, *optional*, defaults to 1) —
  The number of independently computed returned sequences for each element in the batch.
* **output\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more details.
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more details.
* **output\_scores** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
* **output\_logits** (`bool`, *optional*) —
  Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for
  more details.
* **return\_dict\_in\_generate** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput), as opposed to returning exclusively the generated
  sequence. This flag must be set to `True` to return the generation cache (when `use_cache` is `True`)
  or optional outputs (see flags starting with `output_`)

Special tokens that can be used at generation time

* **pad\_token\_id** (`int`, *optional*) —
  The id of the *padding* token.
* **bos\_token\_id** (`int`, *optional*) —
  The id of the *beginning-of-sequence* token.
* **eos\_token\_id** (`Union[int, list[int]]`, *optional*) —
  The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

Generation parameters exclusive to encoder-decoder models

* **encoder\_no\_repeat\_ngram\_size** (`int`, *optional*, defaults to 0) —
  If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
  `decoder_input_ids`.
* **decoder\_start\_token\_id** (`int` or `list[int]`, *optional*) —
  If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length
  `batch_size`. Indicating a list enables different start ids for each element in the batch
  (e.g. multilingual models with different target languages in one batch)

Generation parameters exclusive to assistant generation

* **is\_assistant** (`bool`, *optional*, defaults to `False`) —
  Whether the model is an assistant (draft) model.
* **num\_assistant\_tokens** (`int`, *optional*, defaults to 20) —
  Defines the number of *speculative tokens* that shall be generated by the assistant model before being
  checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation
  more *speculative* : If the assistant model is performant larger speed-ups can be reached, if the assistant
  model requires lots of corrections, lower speed-ups are reached.
* **num\_assistant\_tokens\_schedule** (`str`, *optional*, defaults to `"constant"`) —
  Defines the schedule at which max assistant tokens shall be changed during inference.
  + `"heuristic"`: When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else
    reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
  + `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
  + `"constant"`: `num_assistant_tokens` stays unchanged during generation
* **assistant\_confidence\_threshold** (`float`, *optional*, defaults to 0.4) —
  The confidence threshold for the assistant model. If the assistant model’s confidence in its prediction for the current token is lower
  than this threshold, the assistant model stops the current token generation iteration, even if the number of *speculative tokens*
  (defined by `num_assistant_tokens`) is not yet reached. The assistant’s confidence threshold is adjusted throughout the speculative iterations to reduce the number of unnecessary draft and target forward passes, biased towards avoiding false negatives.
  `assistant_confidence_threshold` value is persistent over multiple generation calls with the same assistant model.
  It is an unsupervised version of the dynamic speculation lookahead
  from Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models <https://huggingface.co/papers/2405.04304>.
* **prompt\_lookup\_num\_tokens** (`int`, *optional*) —
  The number of tokens to be output as candidate tokens.
* **max\_matching\_ngram\_size** (`int`, *optional*) —
  The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.
* **assistant\_early\_exit(`int`,** *optional*) —
  If set to a positive integer, early exit of the model will be used as an assistant. Can only be used with
  models that support early exit (i.e. models where logits from intermediate layers can be interpreted by the LM head).
* **assistant\_lookbehind(`int`,** *optional*, defaults to 10) —
  If set to a positive integer, the re-encodeing process will additionally consider the last `assistant_lookbehind` assistant tokens
  to correctly align tokens. Can only be used with different tokenizers in speculative decoding.
  See this [blog](https://huggingface.co/blog/universal_assisted_generation) for more details.
* **target\_lookbehind(`int`,** *optional*, defaults to 10) —
  If set to a positive integer, the re-encodeing process will additionally consider the last `target_lookbehind` target tokens
  to correctly align tokens. Can only be used with different tokenizers in speculative decoding.
  See this [blog](https://huggingface.co/blog/universal_assisted_generation) for more details.

Parameters related to performances and compilation

* **compile\_config** (CompileConfig, *optional*) —
  If using a compilable cache, this controls how `generate` will `compile` the forward pass for faster
  inference.
* **disable\_compile** (`bool`, *optional*) —
  Whether to disable the automatic compilation of the forward pass. Automatic compilation happens when
  specific criteria are met, including using a compilable cache. Please open an issue if you find the
  need to use this flag.

Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

* *greedy decoding* if `num_beams=1` and `do_sample=False`
* *multinomial sampling* if `num_beams=1` and `do_sample=True`
* *beam-search decoding* if `num_beams>1` and `do_sample=False`
* *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
* *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
* *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
* *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
description of the possible manipulations, as well as examples of their usage.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L837)

( pretrained\_model\_name: typing.Union[str, os.PathLike] config\_file\_name: typing.Union[str, os.PathLike, NoneType] = None cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  ) → [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)

Parameters

* **pretrained\_model\_name** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained model configuration hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a configuration file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig.save_pretrained) method, e.g., `./my_model_directory/`.
* **config\_file\_name** (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`) —
  Name of the generation configuration JSON file to be loaded from `pretrained_model_name`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the configuration files and override the cached versions if
  they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or `bool`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

  To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final configuration object.

  If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused\_kwargs* is a
  dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
  part of `kwargs` which has not been used to update `config` and is otherwise ignored.
* **subfolder** (`str`, *optional*, defaults to `""`) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.
* **kwargs** (`dict[str, Any]`, *optional*) —
  The values in kwargs of any keys which are configuration attributes will be used to override the loaded
  values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
  by the `return_unused_kwargs` keyword parameter.

Returns

[GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)

The configuration object instantiated from this pretrained model.

Instantiate a [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) from a generation configuration file.

Examples:


```
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

#### from\_model\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L1177)

( model\_config: PretrainedConfig  ) → [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)

Parameters

* **model\_config** (`PretrainedConfig`) —
  The model config that will be used to instantiate the generation config.

Returns

[GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)

The configuration object instantiated from those parameters.

Instantiates a [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) from a [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig). This function is useful to convert legacy
[PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) objects, which may contain generation parameters, into a stand-alone [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig).

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L764)

( save\_directory: typing.Union[str, os.PathLike] config\_file\_name: typing.Union[str, os.PathLike, NoneType] = None push\_to\_hub: bool = False \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the configuration JSON file will be saved (will be created if it does not exist).
* **config\_file\_name** (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`) —
  Name of the generation configuration JSON file to be saved in `save_directory`.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig.from_pretrained) class method.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L1221)

( \*\*kwargs  ) → `dict[str, Any]`

Parameters

* **kwargs** (`dict[str, Any]`) —
  Dictionary of attributes to tentatively update this class.

Returns

`dict[str, Any]`

Dictionary containing all the key-value pairs that were not used to update the instance.

Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
returning all the unused kwargs.

#### validate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L544)

( strict = False  )

Parameters

* **strict** (bool) — If True, raise an exception for any issues found. If False, only log issues.

Validates the values of the attributes of the [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) instance. Raises exceptions in the presence
of parameterization that can be detected as incorrect from the configuration instance alone.

Note that some parameters not validated here are best validated at generate runtime, as they may depend on
other inputs and/or the model, such as parameters related to the generation length.

#### get\_generation\_mode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L479)

( assistant\_model: typing.Optional[ForwardRef('PreTrainedModel')] = None  ) → `GenerationMode`

Parameters

* **assistant\_model** (`PreTrainedModel`, *optional*) —
  The assistant model to be used for assisted generation. If set, the generation mode will be
  assisted generation.

Returns

`GenerationMode`

The generation mode triggered by the instance.

Returns the generation mode triggered by the [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) instance.

## GenerationMixin

### class transformers.GenerationMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L350)

( )

A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes.
Inheriting from this class causes the model to have special generation-related behavior, such as loading a
`GenerationConfig` at initialization time or ensuring `generate`-related tests are run in `transformers` CI.

A model class should inherit from `GenerationMixin` to enable calling methods like `generate`, or when it
has defined a custom `generate` method that relies on `GenerationMixin`, directly or indirectly, which
approximately shares the same interface to public methods like `generate`. Three examples:

* `LlamaForCausalLM` should inherit from `GenerationMixin` to enable calling `generate` and other public
  methods in the mixin;
* `BlipForQuestionAnswering` has a custom `generate` method that approximately shares the same interface as
  `GenerationMixin.generate` (it has a few extra arguments, and the same output). That function also calls
  `GenerationMixin.generate` indirectly, through an inner model. As such, `BlipForQuestionAnswering` should
  inherit from `GenerationMixin` to benefit from all generation-related automation in our codebase;
* `BarkModel` has a custom `generate` method and one of its inner models calls `GenerationMixin.generate`.
  However, its `generate` does not share the same interface as `GenerationMixin.generate`. In this case,
  `BarkModel` should NOT inherit from `GenerationMixin`, as it breaks the `generate` interface.

The class exposes [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate), which can be used for:

* *greedy decoding* if `num_beams=1` and `do_sample=False`
* *multinomial sampling* if `num_beams=1` and `do_sample=True`
* *beam-search decoding* if `num_beams>1` and `do_sample=False`
* *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
* *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
* *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
* *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2140)

( inputs: typing.Optional[torch.Tensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None logits\_processor: typing.Optional[transformers.generation.logits\_process.LogitsProcessorList] = None stopping\_criteria: typing.Optional[transformers.generation.stopping\_criteria.StoppingCriteriaList] = None prefix\_allowed\_tokens\_fn: typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None synced\_gpus: typing.Optional[bool] = None assistant\_model: typing.Optional[ForwardRef('PreTrainedModel')] = None streamer: typing.Optional[ForwardRef('BaseStreamer')] = None negative\_prompt\_ids: typing.Optional[torch.Tensor] = None negative\_prompt\_attention\_mask: typing.Optional[torch.Tensor] = None use\_model\_defaults: typing.Optional[bool] = None custom\_generate: typing.Union[str, typing.Callable, NoneType] = None \*\*kwargs  ) → [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

Parameters

* **inputs** (`torch.Tensor` of varying shape depending on the modality, *optional*) —
  The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
  method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
  should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
  `input_ids`, `input_values`, `input_features`, or `pixel_values`.
* **generation\_config** ([GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which has the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complements the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
  sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
  intended for advanced users.
* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) —
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
  `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
  on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
  for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
* **synced\_gpus** (`bool`, *optional*) —
  Whether to continue running the while loop until max\_length. Unless overridden, this flag will be set
  to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
  deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
* **assistant\_model** (`PreTrainedModel`, *optional*) —
  An assistant model that can be used to accelerate generation. The assistant model must have the exact
  same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
  is much faster than running generation with the model you’re calling generate from. As such, the
  assistant model should be much smaller.
* **streamer** (`BaseStreamer`, *optional*) —
  Streamer object that will be used to stream the generated sequences. Generated tokens are passed
  through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
* **negative\_prompt\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  The negative prompt needed for some processors such as CFG. The batch size must match the input batch
  size. This is an experimental feature, subject to breaking API changes in future versions.
* **negative\_prompt\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Attention\_mask for `negative_prompt_ids`.
* **use\_model\_defaults** (`bool`, *optional*) —
  When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
  generation configuration (`model.generation_config`), as opposed to the global defaults
  (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
  `True`.
* **custom\_generate** (`str` or `Callable`, *optional*) —
  One of the following:
  + `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
    `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
    repository fully replaces the generation logic, and the return type may differ.
  + `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
  + `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
    run the decoding loop.
    For more information, see [the docs](../../generation_strategies#custom-generation-methods).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
  specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder\_*.

Returns

[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateDecoderOnlyOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput),
* [GenerateBeamDecoderOnlyOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput)

If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
* [GenerateBeamEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model’s default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](../generation_strategies).

#### compute\_transition\_scores

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L1388)

( sequences: Tensor scores: tuple beam\_indices: typing.Optional[torch.Tensor] = None normalize\_logits: bool = False  ) → `torch.Tensor`

Parameters

* **sequences** (`torch.LongTensor`) —
  The generated sequences. The second dimension (sequence\_length) is either equal to `max_length` or
  shorter if all batches finished early due to the `eos_token_id`.
* **scores** (`tuple(torch.FloatTensor)`) —
  Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
  of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
  Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
  with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
* **beam\_indices** (`torch.LongTensor`, *optional*) —
  Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
  `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
  generate-time.
* **normalize\_logits** (`bool`, *optional*, defaults to `False`) —
  Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

Returns

`torch.Tensor`

A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
the transition scores (logits)

Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
used). This is a convenient method to quickly obtain the scores of the selected tokens at generation time.

Examples:


```
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
>>> output_length = np.sum(transition_scores.numpy() < 0, axis=1)
>>> length_penalty = model.generation_config.length_penalty
>>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
>>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
True
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/text_generation.md)
