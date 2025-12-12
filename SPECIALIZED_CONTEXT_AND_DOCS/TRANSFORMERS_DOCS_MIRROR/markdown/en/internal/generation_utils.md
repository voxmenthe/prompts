# Utilities for Generation

This page lists all the utility functions used by [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate).

## Generate Outputs

The output of [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) is an instance of a subclass of
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput). This output is a data structure containing all the information returned
by [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate), but that can also be used as tuple or dictionary.

Here’s an example:


```
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

The `generation_output` object is a [GenerateDecoderOnlyOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput), as we can
see in the documentation of that class below, it means it has the following attributes:

* `sequences`: the generated sequences of tokens
* `scores` (optional): the prediction scores of the language modelling head, for each generation step
* `hidden_states` (optional): the hidden states of the model, for each generation step
* `attentions` (optional): the attention weights of the model, for each generation step

Here we have the `scores` since we passed along `output_scores=True`, but we don’t have `hidden_states` and
`attentions` because we didn’t pass `output_hidden_states=True` or `output_attentions=True`.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get `None`. Here for instance `generation_output.scores` are all the generated prediction scores of the
language modeling head, and `generation_output.attentions` is `None`.

When using our `generation_output` object as a tuple, it only keeps the attributes that don’t have `None` values.
Here, for instance, it has two elements, `loss` then `logits`, so


```
generation_output[:2]
```

will return the tuple `(generation_output.sequences, generation_output.scores)` for instance.

When using our `generation_output` object as a dictionary, it only keeps the attributes that don’t have `None`
values. Here, for instance, it has two keys that are `sequences` and `scores`.

We document here all output types.

### class transformers.generation.GenerateDecoderOnlyOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L138)

( sequences: LongTensor scores: typing.Optional[tuple[torch.FloatTensor]] = None logits: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None hidden\_states: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None  )

Parameters

* **sequences** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  The generated sequences. The second dimension (sequence\_length) is either equal to `max_length` or shorter
  if all batches finished early due to the `eos_token_id`.
* **scores** (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) —
  Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
  at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
  each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
* **logits** (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) —
  Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
  at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
  each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
* **hidden\_states** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`) —
  Returns the model cache, used to speed up decoding. Different models have a different cache format, check
  the model’s documentation. Usually, a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance.

Outputs of decoder-only generation models, when using non-beam methods.

### class transformers.generation.GenerateEncoderDecoderOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L174)

( sequences: LongTensor scores: typing.Optional[tuple[torch.FloatTensor]] = None logits: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None cross\_attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None decoder\_hidden\_states: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None  )

Parameters

* **sequences** (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`) —
  The generated sequences. The second dimension (sequence\_length) is either equal to `max_length` or shorter
  if all batches finished early due to the `eos_token_id`.
* **scores** (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) —
  Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
  at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
  each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
* **logits** (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) —
  Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
  at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
  each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **decoder\_attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
* **cross\_attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
* **decoder\_hidden\_states** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  Returns the model cache, used to speed up decoding. Different models have a different cache format, check
  the model’s documentation. Usually, a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance.

Outputs of encoder-decoder generation models, when using non-beam methods.

### class transformers.generation.GenerateBeamDecoderOnlyOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L222)

( sequences: LongTensor sequences\_scores: typing.Optional[torch.FloatTensor] = None scores: typing.Optional[tuple[torch.FloatTensor]] = None logits: typing.Optional[tuple[torch.FloatTensor]] = None beam\_indices: typing.Optional[torch.LongTensor] = None attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None hidden\_states: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None  )

Parameters

* **sequences** (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`) —
  The generated sequences. The second dimension (sequence\_length) is either equal to `max_length` or shorter
  if all batches finished early due to the `eos_token_id`.
* **sequences\_scores** (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`) —
  Final beam scores of the generated `sequences`.
* **scores** (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) —
  Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
  of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
  Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
  with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
* **logits** (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) —
  Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
  at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
  each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
* **beam\_indices** (`torch.LongTensor`, *optional*, returned when `output_scores=True`) —
  Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
  `(batch_size*num_return_sequences, sequence_length)`.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
* **hidden\_states** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`) —
  Returns the model cache, used to speed up decoding. Different models have a different cache format, check
  the model’s documentation. Usually, a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance.

Outputs of decoder-only generation models, when using beam methods.

### class transformers.generation.GenerateBeamEncoderDecoderOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L266)

( sequences: LongTensor sequences\_scores: typing.Optional[torch.FloatTensor] = None scores: typing.Optional[tuple[torch.FloatTensor]] = None logits: typing.Optional[tuple[torch.FloatTensor]] = None beam\_indices: typing.Optional[torch.LongTensor] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None cross\_attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None decoder\_hidden\_states: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None  )

Parameters

* **sequences** (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`) —
  The generated sequences. The second dimension (sequence\_length) is either equal to `max_length` or shorter
  if all batches finished early due to the `eos_token_id`.
* **sequences\_scores** (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`) —
  Final beam scores of the generated `sequences`.
* **scores** (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) —
  Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
  of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
  Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
  with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
* **logits** (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) —
  Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
  at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
  each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
* **beam\_indices** (`torch.LongTensor`, *optional*, returned when `output_scores=True`) —
  Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
  `(batch_size*num_return_sequences, sequence_length)`.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
* **decoder\_attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length, sequence_length)`.
* **cross\_attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
* **decoder\_hidden\_states** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) —
  Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
  `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`) —
  Returns the model cache, used to speed up decoding. Different models have a different cache format, check
  the model’s documentation. Usually, a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance.

Outputs of encoder-decoder generation models, when using beam methods.

## LogitsProcessor

A [LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) can be used to modify the prediction scores of a language model head for
generation.

### class transformers.AlternatingCodebooksLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2198)

( input\_start\_len: int semantic\_vocab\_size: int codebook\_size: int  )

Parameters

* **input\_start\_len** (`int`) —
  The length of the initial input sequence.
* **semantic\_vocab\_size** (`int`) —
  Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
* **codebook\_size** (`int`) —
  Number of tokens associated to the codebook.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) enforcing alternated generation between the two codebooks of Bark.

This logits processor is exclusively compatible with
[Bark](https://huggingface.co/docs/transformers/en/model_doc/bark)’s fine submodel. See the model documentation
for examples.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2227)

( input\_ids: LongTensor scores: FloatTensor  )

### class transformers.ClassifierFreeGuidanceLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2134)

( guidance\_scale  )

Parameters

* **guidance\_scale** (float) —
  The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
  Higher guidance scale encourages the model to generate samples that are more closely linked to the input
  prompt, usually at the expense of poorer quality.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) for classifier free guidance (CFG). The scores are split over the batch dimension,
where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
correspond to the unconditional logits (predicted from an empty or ‘null’ prompt). The processor computes a
weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

See [the paper](https://huggingface.co/papers/2306.05284) for more information.

This logits processor is exclusively compatible with
[MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

Examples:


```
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> inputs = processor(
...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2182)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EncoderNoRepeatNGramLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1029)

( encoder\_ngram\_size: int encoder\_input\_ids: LongTensor  )

Parameters

* **encoder\_ngram\_size** (`int`) —
  All ngrams of size `ngram_size` can only occur within the encoder input ids.
* **encoder\_input\_ids** (`int`) —
  The encoder\_input\_ids that should not be repeated within the decoder ids.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that works similarly to [NoRepeatNGramLogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor), but applied exclusively to prevent
the repetition of n-grams present in the prompt.

It was designed to promote chattiness in a language model, by preventing the generation of n-grams present in
previous conversation rounds.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

>>> inputs = tokenizer("Alice: I love cats. What do you love?\nBob:", return_tensors="pt")

>>> # With greedy decoding, we see Bob repeating Alice's opinion. If Bob was a chatbot, it would be a poor one.
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
Alice: I love cats. What do you love?
Bob: I love cats. What do you

>>> # With this logits processor, we can prevent Bob from repeating Alice's opinion.
>>> outputs = model.generate(**inputs, encoder_no_repeat_ngram_size=2)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
Alice: I love cats. What do you love?
Bob: My cats are very cute.
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1078)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EncoderRepetitionPenaltyLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L410)

( penalty: float encoder\_input\_ids: LongTensor  )

Parameters

* **penalty** (`float`) —
  The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0
  and 1.0 penalizes prompt tokens.
* **encoder\_input\_ids** (`torch.LongTensor`) —
  The encoder\_input\_ids that should be repeated within the decoder ids.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that works similarly to [RepetitionPenaltyLogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.RepetitionPenaltyLogitsProcessor), but with an *inverse* penalty
that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
selecting tokens that were present in the prompt.

It was designed to avoid hallucination in input-grounded tasks, like summarization. Although originally intended
for encoder-decoder models, it can also be used with decoder-only models like LLMs.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
>>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

>>> inputs = tokenizer(["Alice and Bob. The third member's name was"], return_tensors="pt")
>>> gen_out = model.generate(**inputs)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
Alice and Bob. The third member's name was not mentioned.

>>> # With the `encoder_repetition_penalty` argument we can trigger this logits processor in `generate`, which can
>>> # promote the use of prompt tokens ("Bob" in this example)
>>> gen_out = model.generate(**inputs, encoder_repetition_penalty=1.2)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
Alice and Bob. The third member's name was Bob. The third member's name was Bob.
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L454)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EpsilonLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L750)

( epsilon: float filter\_value: float = -inf min\_tokens\_to\_keep: int = 1  )

Parameters

* **epsilon** (`float`) —
  If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
* **filter\_value** (`float`, *optional*, defaults to -inf) —
  All filtered values will be set to this float value.
* **min\_tokens\_to\_keep** (`int`, *optional*, defaults to 1) —
  Minimum number of tokens that cannot be filtered.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
largest min\_tokens\_to\_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
Desmoothing](https://huggingface.co/papers/2210.15191) for more information.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
<BLANKLINE>
<BLANKLINE>

>>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
>>> # Top P sampling, which restricts tokens based on their cumulative probability.
>>> # Pro tip: The paper recommends using `epsilon_cutoff` values between 3e-4 and 9e-4
>>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L805)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EtaLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L819)

( epsilon: float filter\_value: float = -inf min\_tokens\_to\_keep: int = 1 device: str = 'cpu'  )

Parameters

* **epsilon** (`float`) —
  A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The
  suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
* **filter\_value** (`float`, *optional*, defaults to -inf) —
  All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This
  parameter is useful when logits need to be modified for very low probability tokens that should be excluded
  from generation entirely.
* **min\_tokens\_to\_keep** (`int`, *optional*, defaults to 1) —
  Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities.
  For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation,
  even if all tokens have probabilities below the cutoff `eta`.
* **device** (`str`, *optional*, defaults to `"cpu"`) —
  The device to allocate the tensors.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
min\_tokens\_to\_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more information. Note: `do_sample`
must be set to `True` for this `LogitsProcessor` to work.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
<BLANKLINE>
<BLANKLINE>

>>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
>>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
>>> # Pro tip: The paper recommends using `eta_cutoff` values between 3e-4 to 4e-3
>>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L888)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ExponentialDecayLengthPenalty

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1702)

( exponential\_decay\_length\_penalty: tuple eos\_token\_id: typing.Union[int, list[int], torch.Tensor] input\_ids\_seq\_length: int  )

Parameters

* **exponential\_decay\_length\_penalty** (`tuple(int, float)`) —
  This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
  starts and `decay_factor` represents the factor of exponential decay
* **eos\_token\_id** (`Union[int, list[int], torch.Tensor]`) —
  The id(s) of the *end-of-sequence* token.
* **input\_ids\_seq\_length** (`int`) —
  The length of the input sequence.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that exponentially increases the score of the `eos_token_id` after `start_index` has been
reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
predicted in a meaningful position.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

>>> text = "Just wanted to let you know, I"
>>> inputs = tokenizer(text, return_tensors="pt")

>>> # Let's consider that we want short sentences, so we limit `max_length=30`. However, we observe that the answer
>>> # tends to end abruptly.
>>> set_seed(1)
>>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
published in 2010. Although

>>> # To promote the appearance of the EOS token at the right time, we add the `exponential_decay_length_penalty =
>>> # (start_index, decay_factor)`. Instead of cutting at max_tokens, the output comes to an end before and usually
>>> # with more meaning. What happens is that starting from `start_index` the EOS token score will be increased
>>> # by `decay_factor` exponentially. However, if you set a high decay factor, you may also end up with abruptly
>>> # ending sequences.
>>> set_seed(1)
>>> outputs = model.generate(
...     **inputs,
...     do_sample=True,
...     temperature=0.9,
...     max_length=30,
...     pad_token_id=50256,
...     exponential_decay_length_penalty=(15, 1.6),
... )
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network
which<|endoftext|>

>>> # With a small decay factor, you will have a higher chance of getting a meaningful sequence.
>>> set_seed(1)
>>> outputs = model.generate(
...     **inputs,
...     do_sample=True,
...     temperature=0.9,
...     max_length=30,
...     pad_token_id=50256,
...     exponential_decay_length_penalty=(15, 1.01),
... )
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
published in 2010.<|endoftext|>
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1788)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ForcedBOSTokenLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1580)

( bos\_token\_id: int  )

Parameters

* **bos\_token\_id** (`int`) —
  The id of the token to force as the first generated token.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that enforces the specified token as the first generated token. Used with encoder-decoder
models.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

>>> inputs = tokenizer("Translate from English to German: I love cats.", return_tensors="pt")

>>> # By default, it continues generating according to the model's logits
>>> outputs = model.generate(**inputs, max_new_tokens=10)
>>> print(tokenizer.batch_decode(outputs)[0])
<pad> Ich liebe Kitty.</s>

>>> # We can use `forced_bos_token_id` to force the start of generation with an encoder-decoder model
>>> # (including forcing it to end straight away with an EOS token)
>>> outputs = model.generate(**inputs, max_new_tokens=10, forced_bos_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(outputs)[0])
<pad></s>
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1615)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ForcedEOSTokenLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1625)

( max\_length: int eos\_token\_id: typing.Union[int, list[int], torch.Tensor] device: str = 'cpu'  )

Parameters

* **max\_length** (`int`) —
  The maximum length of the sequence to be generated.
* **eos\_token\_id** (`Union[int, list[int], torch.Tensor]`) —
  The id(s) of the *end-of-sequence* token.
* **device** (`str`, *optional*, defaults to `"cpu"`) —
  The device to allocate the tensors.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that enforces the specified token as the last generated token when `max_length` is reached.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

>>> # By default, it continues generating according to the model's logits
>>> outputs = model.generate(**inputs, max_new_tokens=10)
>>> print(tokenizer.batch_decode(outputs)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8

>>> # `forced_eos_token_id` ensures the generation ends with a EOS token
>>> outputs = model.generate(**inputs, max_new_tokens=10, forced_eos_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(outputs)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7,<|endoftext|>
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1671)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.HammingDiversityLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1444)

( diversity\_penalty: float num\_beams: int num\_beam\_groups: int  )

Parameters

* **diversity\_penalty** (`float`) —
  This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a
  particular time. A higher `diversity_penalty` will enforce greater diversity among the beams. Adjusting
  this value can help strike a balance between diversity and natural likelihood.
* **num\_beams** (`int`) —
  Number of beams for beam search. 1 means no beam search.
* **num\_beam\_groups** (`int`) —
  Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
  [this paper](https://huggingface.co/papers/1610.02424) for more details.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that enforces diverse beam search.

Note that this logits processor is only effective for `PreTrainedModel.group_beam_search`. See [Diverse Beam
Search: Decoding Diverse Solutions from Neural Sequence Models](https://huggingface.co/papers/1610.02424) for more
details.

Traditional beam search often generates very similar sequences across different beams.
`HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
beams in the same time step.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> # Initialize the model and tokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

>>> # A long text about the solar system
>>> text = (
...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
...     "interstellar molecular cloud."
... )
>>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

>>> # Generate diverse summary
>>> outputs_diverse = model.generate(
...     **inputs,
...     num_beam_groups=2,
...     diversity_penalty=10.0,
...     max_length=100,
...     num_beams=4,
...     num_return_sequences=2,
... )
>>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

>>> # Generate non-diverse summary
>>> outputs_non_diverse = model.generate(
...     **inputs,
...     max_length=100,
...     num_beams=4,
...     num_return_sequences=2,
... )
>>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

>>> # With `diversity_penalty`, the resulting beams are much more diverse
>>> print(summary_non_diverse)
['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

>>> print(summaries_diverse)
['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1531)

( input\_ids: LongTensor scores: FloatTensor current\_tokens: LongTensor beam\_group\_idx: int  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
  beam search or log softmax for each vocabulary token when using beam search
* **current\_tokens** (`torch.LongTensor` of shape `(batch_size)`) —
  Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
  beam groups in the current generation step.
* **beam\_group\_idx** (`int`) —
  The index of the beam group currently being processed.

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.InfNanRemoveLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1681)

( )

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
the logits processor should only be used if necessary since it can slow down the generation method.

This logits processor has no `generate` example, as there shouldn’t be a correct combination of flags that warrants
its use.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1690)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.LogitNormalization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1803)

( )

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) for normalizing the scores using log-softmax. It’s important to normalize
the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
this library doesn’t do it (it only does it before, but they may need re-normalization) but it still supposes that
the scores are normalized when comparing the hypotheses.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> import torch

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

>>> # By default, the scores are not normalized -- the sum of their exponentials is NOT a normalized probability
>>> # distribution, summing to 1
>>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
>>> print(torch.allclose(torch.sum(torch.exp(outputs.scores[-1])), torch.Tensor((1.000,)), rtol=1e-4))
False

>>> # Normalizing them may have a positive impact on beam methods, or when using the scores on your application
>>> outputs = model.generate(**inputs, renormalize_logits=True, return_dict_in_generate=True, output_scores=True)
>>> print(torch.allclose(torch.sum(torch.exp(outputs.scores[-1])), torch.Tensor((1.000,)), rtol=1e-4))
True
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1834)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.LogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L50)

( )

Abstract base class for all logit processors that can be applied during generation.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L53)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.LogitsProcessorList

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L60)

( iterable = ()  )

This class can be used to create a list of [LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) to subsequently process a `scores` input tensor.
This class inherits from list and adds a specific ***call*** method to apply each [LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) to the
inputs.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L67)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
  beam search or log softmax for each vocabulary token when using beam search
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional kwargs that are specific to a logits processor.

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.MinLengthLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L98)

( min\_length: int eos\_token\_id: typing.Union[int, list[int], torch.Tensor] device: str = 'cpu'  )

Parameters

* **min\_length** (`int`) —
  The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
* **eos\_token\_id** (`Union[int, list[int], torch.Tensor]`) —
  The id(s) of the *end-of-sequence* token.
* **device** (`str`, *optional*, defaults to `"cpu"`) —
  The device to allocate the tensors.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
like most LLMs, the length includes the prompt.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
>>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

>>> inputs = tokenizer("A number:", return_tensors="pt")
>>> gen_out = model.generate(**inputs)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
A number: one

>>> # setting `min_length` to a value smaller than the uncontrolled output length has no impact
>>> gen_out = model.generate(**inputs, min_length=3)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
A number: one

>>> # setting a larger `min_length` will force the model to generate beyond its natural ending point, which is not
>>> # necessarily incorrect
>>> gen_out = model.generate(**inputs, min_length=10)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
A number: one thousand, nine hundred and ninety-four
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L149)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.MinNewTokensLengthLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L159)

( prompt\_length\_to\_skip: int min\_new\_tokens: int eos\_token\_id: typing.Union[int, list[int], torch.Tensor] device: str = 'cpu'  )

Parameters

* **prompt\_length\_to\_skip** (`int`) —
  The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
  input length.
* **min\_new\_tokens** (`int`) —
  The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
* **eos\_token\_id** (`Union[int, list[int], torch.Tensor]`) —
  The id(s) of the *end-of-sequence* token.
* **device** (`str`, *optional*, defaults to `"cpu"`) —
  The device to allocate the tensors.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
Contrarily to [MinLengthLogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.MinLengthLogitsProcessor), this processor ignores the prompt.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
>>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

>>> inputs = tokenizer(["A number:"], return_tensors="pt")
>>> gen_out = model.generate(**inputs)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
A number: one

>>> # setting `min_new_tokens` will force the model to generate beyond its natural ending point, which is not
>>> # necessarily incorrect
>>> gen_out = model.generate(**inputs, min_new_tokens=2)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
A number: one thousand
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L219)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.MinPLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L585)

( min\_p: float filter\_value: float = -inf min\_tokens\_to\_keep: int = 1  )

Parameters

* **min\_p** (`float`) —
  Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
  value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
  the 0.99-0.8 range (use the opposite of normal `top_p` values).
* **filter\_value** (`float`, *optional*, defaults to -inf) —
  All filtered values will be set to this float value.
* **min\_tokens\_to\_keep** (`int`, *optional*, defaults to 1) —
  Minimum number of tokens that cannot be filtered.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that performs min-p, i.e. keeps all tokens that are above a minimum probability, scaled by the
probability of the most likely token. As a result, the filter becomes more aggressive in the presence of
high-probability tokens, which is a sign of a confident output that we shouldn’t deviate from.

Often used together with [TemperatureLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper). Used as an alternative to [TopPLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TopPLogitsWarper) and
[TopKLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TopKLogitsWarper).

Created by @menhguin and @kalomaze (github handles). Code adapted from [this external PR](https://github.com/oobabooga/text-generation-webui/pull/4449/files)

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
<BLANKLINE>
<BLANKLINE>

>>> # With `min_p` sampling, the output gets restricted to high-probability tokens.
>>> # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
>>> outputs = model.generate(**inputs, do_sample=True, min_p=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L642)

( input\_ids: LongTensor scores: FloatTensor  )

### class transformers.NoBadWordsLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1281)

( bad\_words\_ids: list eos\_token\_id: typing.Union[int, list[int], torch.Tensor, NoneType] = None  )

Parameters

* **bad\_words\_ids** (`list[list[int]]`) —
  List of list of token ids that are not allowed to be generated.
* **eos\_token\_id** (`Union[int, list[int], torch.Tensor]`, *optional*) —
  The id(s) of the *end-of-sequence* token.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that enforces that specified sequences will never be selected.

In order to get the token ids of the words that should not appear in the generated text, make sure to set
`add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
as fast tokenizers’ prefixing behaviours come from `pre tokenizers`. Read more
[here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

>>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a mess.

>>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
>>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


>>> def get_tokens_as_list(word_list):
...     "Converts a sequence of words into a list of tokens"
...     tokens_list = []
...     for word in word_list:
...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
...         tokens_list.append(tokenized_word)
...     return tokens_list


>>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
>>> output_ids = model.generate(
...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
... )
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a surprise.
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1173)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.NoRepeatNGramLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L970)

( ngram\_size: int  )

Parameters

* **ngram\_size** (`int`) —
  All ngrams of size `ngram_size` can only occur once.

N-grams are groups of “n” consecutive words, characters, or tokens taken from a sequence of text. Given the
sentence: “She runs fast”, the bi-grams (n=2) would be (“she”, “runs”) and (“runs”, “fast”). In text generation,
avoiding repetitions of word sequences provides a more diverse output. This [LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) enforces no
repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
from consideration when further processing the scores. Note that, for decoder-only models like most LLMs, the
prompt is also considered to obtain the n-grams.
[Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
might lead to undesirable outcomes where the city’s name appears only once in the entire text.
[Reference](https://huggingface.co/blog/how-to-generate)

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
>>> inputs = tokenizer(["Today I"], return_tensors="pt")

>>> output = model.generate(**inputs)
>>> print(tokenizer.decode(output[0], skip_special_tokens=True))
Today I'm not sure if I'm going to be able to do it.

>>> # Now let's add ngram size using `no_repeat_ngram_size`. This stops the repetitions ("I'm") in the output.
>>> output = model.generate(**inputs, no_repeat_ngram_size=2)
>>> print(tokenizer.decode(output[0], skip_special_tokens=True))
Today I'm not sure if I can get a better understanding of the nature of this issue
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1017)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.PrefixConstrainedLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1372)

( prefix\_allowed\_tokens\_fn: typing.Callable[[int, torch.Tensor], list[int]] num\_beams: int  )

Parameters

* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`) —
  This function constraints the beam search to allowed tokens only at each step. This function takes 2
  arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
  next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
  `batch_id`.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that enforces constrained generation and is useful for prefix-conditioned constrained
generation. See [Autoregressive Entity Retrieval](https://huggingface.co/papers/2010.00904) for more information.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

>>> inputs = tokenizer("Alice and Bob", return_tensors="pt")

>>> # By default, it continues generating according to the model's logits
>>> outputs = model.generate(**inputs, max_new_tokens=5)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
Alice and Bob are friends

>>> # We can constrain it with `prefix_allowed_tokens_fn` to force a certain behavior based on a prefix.
>>> # For instance, we can force an entire entity to be generated when its beginning is detected.
>>> entity = tokenizer(" Bob Marley", return_tensors="pt").input_ids[0]  # 3 tokens
>>> def prefix_allowed_tokens_fn(batch_id, input_ids):
...     '''
...     Attempts to generate 'Bob Marley' when 'Bob' is detected.
...     In this case, `batch_id` is not used, but you can set rules for each batch member.
...     '''
...     if input_ids[-1] == entity[0]:
...         return [entity[1].item()]
...     elif input_ids[-2] == entity[0] and input_ids[-1] == entity[1]:
...         return [entity[2].item()]
...     return list(range(tokenizer.vocab_size))  # If no match, allow all tokens

>>> outputs = model.generate(**inputs, max_new_tokens=5, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
Alice and Bob Marley
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1423)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.RepetitionPenaltyLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L297)

( penalty: float prompt\_ignore\_length: typing.Optional[int] = None  )

Parameters

* **penalty** (`float`) —
  The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated
  tokens. Between 0.0 and 1.0 rewards previously generated tokens.
* **prompt\_ignore\_length** (`int`, *optional*) —
  The original input ids sequence length, which if provided, will not be used in the penalty calculation.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that prevents the repetition of previous tokens through a penalty. This penalty is applied at
most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt
by default.

In the original [paper](https://huggingface.co/papers/1909.05858), the authors suggest the use of a penalty of around
1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, RepetitionPenaltyLogitsProcessor

>>> # Initializing the model and tokenizer for it
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
>>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

>>> # This shows a normal generate without any specific parameters
>>> summary_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'm going to be able to do that

>>> # This generates a penalty for repeated tokens
>>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
>>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'll just have to go out and play

>>> # We can also exclude the input prompt by creating an instance of this class
>>> # with a `prompt_ignore_length` and passing it as a custom logit processor
>>> rep_pen_processor = RepetitionPenaltyLogitsProcessor(
...     penalty=1.1,
...     prompt_ignore_length=inputs["input_ids"].shape[-1]
... )
>>> penalized_ids = model.generate(**inputs, logits_processor=[rep_pen_processor])
>>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'm going to have to go through a lot of things, and
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L365)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SequenceBiasLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1098)

( sequence\_bias: list  )

Parameters

* **sequence\_bias** (`list[list[Union[list[int], float]]]`) —
  List of lists that maps a sequence of tokens to its bias term (e.g. `[[[10, 45], -2.0], [[64], -7.5]]`). Positive biases increase the odds of the
  sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
  will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
  completed (in the token selection step after this processor is applied).

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that applies an additive bias on sequences. The bias is applied to the last token of a sequence
when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
one token, consider using beam methods (to gracefully work around partially completed sequences that have a
negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

At a token-level, biasing a word is different from biasing a word with a space before it. If you want to bias
“foo” mid-sentence, you’ll likely want to add a prefix space and bias ” foo” instead. Check the tokenizer section
of our NLP course to find out why: <https://huggingface.co/learn/nlp-course/chapter2/4?fw=pt>

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
>>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

>>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4, do_sample=False)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald John Trump Sr.

>>> def get_tokens(word):
...     return tokenizer([word], add_special_tokens=False).input_ids[0]

>>> # IMPORTANT: Remember our tip about adding spaces before words to bias them correctly.
>>> sequence_bias = [[get_tokens("Trump"), -10.0],]  # will fail to apply bias
>>> biased_ids = model.generate(
...     inputs["input_ids"], max_new_tokens=4, do_sample=False, sequence_bias=sequence_bias
... )
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald John Trump Sr.

>>> sequence_bias = [[get_tokens(" Trump"), -10.0],]  # will work
>>> biased_ids = model.generate(
...     inputs["input_ids"], max_new_tokens=4, do_sample=False, sequence_bias=sequence_bias
... )
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald John Harper. He

>>> # We can also add a positive bias to nudge the model towards specific tokens or continuations. This technique
>>> # is also more effective when paired up with beam search.
>>> sequence_bias = [[get_tokens(" Donald Duck"), 10.0],]
>>> biased_ids = model.generate(
...     inputs["input_ids"], max_new_tokens=4, num_beams=4, do_sample=False, sequence_bias=sequence_bias
... )
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald Duck. He is
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1173)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SuppressTokensAtBeginLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1840)

( begin\_suppress\_tokens begin\_index device: str = 'cpu'  )

[SuppressTokensAtBeginLogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.SuppressTokensAtBeginLogitsProcessor) suppresses a list of tokens as soon as the `generate` function starts
generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
not generated at the beginning. Originally created for
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

Examples:


```
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

>>> # Whisper has `begin_suppress_tokens` set by default (= `[220, 50256]`). 50256 is the EOS token, so this means
>>> # it can't generate and EOS token in the first iteration, but it can in the others.
>>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
>>> print(outputs.scores[0][0, 50256])
tensor(-inf)
>>> print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS
tensor(29.9010)

>>> # If we disable `begin_suppress_tokens`, we can generate EOS in the first iteration.
>>> outputs = model.generate(
...     **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
... )
>>> print(outputs.scores[0][0, 50256])
tensor(11.2027)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1882)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SuppressTokensLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1893)

( suppress\_tokens device: str = 'cpu'  )

This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so
that they are not generated. Originally created for
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

Examples:


```
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

>>> # Whisper has a long list of suppressed tokens. For instance, in this case, the token 1 is suppressed by default.
>>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
>>> print(outputs.scores[1][0, 1])  # 1 (and not 0) is the first freely generated token
tensor(-inf)

>>> # If we disable `suppress_tokens`, we can generate it.
>>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, suppress_tokens=None)
>>> print(outputs.scores[1][0, 1])
tensor(6.0678)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1925)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SynthIDTextWatermarkLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2581)

( ngram\_len: int keys: list sampling\_table\_size: int sampling\_table\_seed: int context\_history\_size: int device: device skip\_first\_ngram\_calls: bool = False debug\_mode: bool = False  )

Parameters

* **ngram\_len** (`int`) —
  Ngram length.
* **keys** (`list[int]`) —
  A sequence of watermarking keys, one for each depth.
* **sampling\_table\_size** (`int`) —
  Size of the sampling table.
* **sampling\_table\_seed** (`int`) —
  Random seed to generate the sampling table.
* **context\_history\_size** (`int`) —
  Size of the tensor to keep track of seen contexts.
* **device** (`torch.device`) —
  Device to use.
* **skip\_first\_ngram\_calls** (`bool`, *optional*, defaults to `False`) —
  Whether to skip first ngram calls.
* **debug\_mode** (`bool`, optional, *optional*, defaults to `False`) —
  Logits are modified to uniform one got before watermarking modification is applied. This is to test the
  implementation.

Logits processor that implements watermarking techniques for text generation models.
This class facilitates the application of SynthID text watermarking, a method for embedding imperceptible signals
into generated text to aid in detecting synthetic content. It operates by subtly manipulating the probabilities of
token selection during text generation in a manner that can be reliably recovered later for verification.

Key Features:

* **State Management:** Maintains internal state to track token sequences and generate watermarking keys
  dynamically.
* **Key Generation:** Computes hashes based on token sequences and watermarking parameters to create unique keys
  for each position.
* **G-Value Sampling:** Employs a pre-computed sampling table to sample watermarking values (g-values) based on
  the generated keys.
* **Score Adjustment:** Applies calculated g-values to modify token probabilities during generation, embedding the
  watermark.
* **Context Repetition Handling:** Incorporates logic to avoid watermarking tokens in repeated contexts,
  preserving naturalness.
* **EOS Token Masking:** Supports masking end-of-sentence tokens to prevent their inclusion in watermarking
  calculations.
* **Utility Functions:** Provides functions to compute g-values directly, check for context repetition, create
  EOS token masks, and estimate expected mean g-values.

Refer to paper url: <https://www.nature.com/articles/s41586-024-08025-4> for more details around this.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

>>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
>>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

>>> # SynthID Text configuration
>>> watermarking_config = SynthIDTextWatermarkingConfig(
...     keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
...     ngram_len=5,
... )

>>> # Generation with watermarking
>>> tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
>>> output_sequences = model.generate(
...     **tokenized_prompts, watermarking_config=watermarking_config, do_sample=True, max_new_tokens=10
... )
>>> watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2719)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TemperatureLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L231)

( temperature: float  )

Parameters

* **temperature** (`float`) —
  Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
  randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
  token.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) for temperature (exponential scaling output probability distribution), which effectively means
that it can control the randomness of the predicted tokens. Often used together with [TopPLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TopPLogitsWarper) and
[TopKLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TopKLogitsWarper).

Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won’t have
any effect.

Examples:


```
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)  # for reproducibility

>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> model.config.pad_token_id = model.config.eos_token_id
>>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

>>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
>>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
>>> outputs = model.generate(**inputs, **generate_kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Hugging Face Company is one of these companies that is going to take a',
"Hugging Face Company is a brand created by Brian A. O'Neil"]

>>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
>>> generate_kwargs["temperature"] = 0.0001
>>> outputs = model.generate(**inputs, **generate_kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Hugging Face Company is a company that has been around for over 20 years',
'Hugging Face Company is a company that has been around for over 20 years']
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L291)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TopKLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L532)

( top\_k: int filter\_value: float = -inf min\_tokens\_to\_keep: int = 1  )

Parameters

* **top\_k** (`int`) —
  The number of highest probability vocabulary tokens to keep for top-k-filtering.
* **filter\_value** (`float`, *optional*, defaults to -inf) —
  All filtered values will be set to this float value.
* **min\_tokens\_to\_keep** (`int`, *optional*, defaults to 1) —
  Minimum number of tokens that cannot be filtered.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that performs top-k, i.e. restricting to the k highest probability elements. Often used
together with [TemperatureLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper) and [TopPLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TopPLogitsWarper).

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: A, B, C, D", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: A, B, C, D, E — S — O, P — R

>>> # With `top_k` sampling, the output gets restricted the k most likely tokens.
>>> # Pro tip: In practice, LLMs use `top_k` in the 5-50 range.
>>> outputs = model.generate(**inputs, do_sample=True, top_k=2)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: A, B, C, D, E, F, G, H, I
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L576)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TopPLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L465)

( top\_p: float filter\_value: float = -inf min\_tokens\_to\_keep: int = 1  )

Parameters

* **top\_p** (`float`) —
  If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
  higher are kept for generation.
* **filter\_value** (`float`, *optional*, defaults to -inf) —
  All filtered values will be set to this float value.
* **min\_tokens\_to\_keep** (`int`, *optional*, defaults to 1) —
  Minimum number of tokens that cannot be filtered.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that performs top-p, i.e. restricting to top tokens summing to prob\_cut\_off <= prob\_cut\_off.
Often used together with [TemperatureLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper) and [TopKLogitsWarper](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.TopKLogitsWarper).

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
<BLANKLINE>
<BLANKLINE>

>>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
>>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
>>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L516)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TypicalLogitsWarper

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L662)

( mass: float = 0.9 filter\_value: float = -inf min\_tokens\_to\_keep: int = 1  )

Parameters

* **mass** (`float`, *optional*, defaults to 0.9) —
  Value of typical\_p between 0 and 1 inclusive, defaults to 0.9.
* **filter\_value** (`float`, *optional*, defaults to -inf) —
  All filtered values will be set to this float value.
* **min\_tokens\_to\_keep** (`int`, *optional*, defaults to 1) —
  Minimum number of tokens that cannot be filtered.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that performs typical decoding. Inspired on how humans use language, it prioritizes tokens
whose log probability is close to the entropy of the token probability distribution. This means that the most
likely tokens may be discarded in the process.

See [Typical Decoding for Natural Language Generation](https://huggingface.co/papers/2202.00666) for more information.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
>>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

>>> inputs = tokenizer("1, 2, 3", return_tensors="pt")

>>> # We can see that greedy decoding produces a sequence of numbers
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

>>> # For this particular seed, we can see that sampling produces nearly the same low-information (= low entropy)
>>> # sequence
>>> set_seed(18)
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
1, 2, 3, 4, 5, 6, 7, 8, 9 and 10

>>> # With `typical_p` set, the most obvious sequence is no longer produced, which may be good for your problem
>>> set_seed(18)
>>> outputs = model.generate(
...     **inputs, do_sample=True, typical_p=0.1, return_dict_in_generate=True, output_scores=True
... )
>>> print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])
1, 2, 3 and 5

>>> # We can see that the token corresponding to "4" (token 934) in the second position, the most likely token
>>> # as seen with greedy decoding, was entirely blocked out
>>> print(outputs.scores[1][0, 934])
tensor(-inf)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L726)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2243)

( guidance\_scale: float model unconditional\_ids: typing.Optional[torch.LongTensor] = None unconditional\_attention\_mask: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = True  )

Parameters

* **guidance\_scale** (`float`) —
  The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
  Higher guidance scale encourages the model to generate samples that are more closely linked to the input
  prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
  making the negative prompt provided with negative\_prompt\_ids (if any) act as a positive prompt.
* **model** (`PreTrainedModel`) —
  The model computing the unconditional scores. Supposedly the same as the one computing the conditional
  scores. Both models must use the same tokenizer.
* **unconditional\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
  the last token of the prompt.
* **unconditional\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Attention mask for unconditional\_ids.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether to cache key/values during the negative prompt forward pass.

Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

See [the paper](https://huggingface.co/papers/2306.17806) for more information.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
>>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

>>> # with a negative prompt
>>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
>>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

>>> # with a positive prompt
>>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
>>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2349)

( input\_ids scores  )

### class transformers.WhisperTimeStampLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L1933)

( generate\_config: GenerationConfig begin\_index: int \_detect\_timestamp\_from\_logprob: typing.Optional[bool] = None  )

Parameters

* **generate\_config** (`GenerateConfig`) —
  The generate config used to generate the output. The following parameters are required:
  eos\_token\_id (`int`, *optional*, defaults to 50257):
  The id of the *end-of-sequence* token.
  no\_timestamps\_token\_id (`int`, *optional*, defaults to 50363):
  The id of the `"<|notimestamps|>"` token.
  max\_initial\_timestamp\_index (`int`, *optional*, defaults to 1):
  Used to set the maximum value of the initial timestamp. This is used to prevent the model from
  predicting timestamps that are too far in the future.
* **begin\_index** (`int`) —
  Token index of the first token that is generated by the model.
* **\_detect\_timestamp\_from\_logprob** (`bool`, *optional*) —
  Whether timestamps can be predicted from logprobs over all timestamps.

[LogitsProcessor](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.LogitsProcessor) that modifies the logits for the generation of timestamps in the transcription. When the input
tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
potential tokens.

See [the paper](https://huggingface.co/papers/2212.04356) for more information.

Examples:


```
>>> import torch
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration, GenerationConfig
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features

>>> #Displaying timestamps
>>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
>>> transcription = processor.batch_decode(generated_ids, decode_with_timestamps=True)[0]
>>> print("Transcription:", transcription)
Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>


>>> #No timestamps & change EOS:
>>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
>>> model.generation_config.eos_token_id = 460
>>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print("Transcription:", transcription)
Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2022)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.WatermarkLogitsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2408)

( vocab\_size device greenlist\_ratio: float = 0.25 bias: float = 2.0 hashing\_key: int = 15485863 seeding\_scheme: str = 'lefthash' context\_width: int = 1  )

Parameters

* **vocab\_size** (`int`) —
  The model tokenizer’s vocab\_size. Used to calculate “green” tokens ratio.
* **device** (`str`) —
  The device where model is allocated.
* **greenlist\_ratio** (`float`, optional, *optional*, defaults to 0.25) —
  The ratio of “green” tokens used to the vocabulary size. Defaults to 0.25.
* **bias** (`float`, optional, *optional*, defaults to 2.0) —
  The bias added to the selected “green” tokens’ logits. Consider lowering the
  `bias` if the text generation quality degrades. Recommended values are in the
  range of [0.5, 2.0]. Defaults to 2.0.
* **hashing\_key** (`int`, optional, *optional*, defaults to 15485863) —
  Key used for hashing. If you deploy this watermark, we advise using another private key.
  Defaults to 15485863 (the millionth prime).
* **seeding\_scheme** (`str`, optional, *optional*, defaults to `"lefthash"`) —
  The seeding scheme used for selecting “green” tokens. Accepts values:
  + “lefthash” (default): “green” tokens selection depend on the last token (Algorithm 2 from paper)
  + “selfhash”: “green” tokens selection depends on the current token itself (Algorithm 3 from paper)
    The downside of this scheme is that it considers all possible next tokens and can be slower than “lefthash”.
    The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
* **context\_width** (`int`, *optional*, defaults to 1) —
  The number of previous tokens to use when setting the seed.

Logits processor for watermarking generated text. The processor modifies model output scores by adding a small bias to
randomized set of “green” tokens before generating the next token. “Green” tokens selection process depends on the
`seeding_scheme` used. The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

The text generated by this `LogitsProcessor` can be detected using `WatermarkDetector`. See [**call**()](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.WatermarkDetector.__call__) for details,

See [the paper](https://huggingface.co/papers/2306.04634) for more information.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig

>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> inputs = tokenizer(["Alice and Bob are"], return_tensors="pt")

>>> # normal generation
>>> out = model.generate(inputs["input_ids"], max_length=20, do_sample=False)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Alice and Bob are both in the same room.\n\n"I\'m not sure if you\'re'

>>> # watermarked generation
>>> watermarking_config = WatermarkingConfig(bias=2.5, context_width=2, seeding_scheme="selfhash")
>>> out = model.generate(inputs["input_ids"], watermarking_config=watermarking_config, max_length=20, do_sample=False)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Alice and Bob are both still alive and well and the story is pretty much a one-hour adventure'

>>> # to detect watermarked text use the WatermarkDetector class
>>> from transformers import WatermarkDetector
>>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config= watermarking_config)
>>> detection_preds = detector(out)
>>> detection_preds
array([ True])
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/logits_process.py#L2530)

( input\_ids: LongTensor scores: FloatTensor  ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

## StoppingCriteria

A [StoppingCriteria](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.StoppingCriteria) can be used to change when to stop generation (other than EOS token). Please note that this is exclusively available to our PyTorch implementations.

### class transformers.StoppingCriteria

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L47)

( )

Abstract base class for all stopping criteria that can be applied during generation.

If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L54)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional stopping criteria specific kwargs.

Returns

`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

### class transformers.StoppingCriteriaList

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L497)

( iterable = ()  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L498)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional stopping criteria specific kwargs.

Returns

`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

### class transformers.MaxLengthCriteria

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L59)

( max\_length: int max\_position\_embeddings: typing.Optional[int] = None  )

Parameters

* **max\_length** (`int`) —
  The maximum length that the output sequence can have in number of tokens.
* **max\_position\_embeddings** (`int`, *optional*) —
  The maximum model length, as defined by the model’s `config.max_position_embeddings` attribute.

This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
in mind for decoder-only type of transformers, this will include the initial prompted tokens.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L75)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional stopping criteria specific kwargs.

Returns

`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

### class transformers.MaxTimeCriteria

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L88)

( max\_time: float initial\_timestamp: typing.Optional[float] = None  )

Parameters

* **max\_time** (`float`) —
  The maximum allowed time in seconds for the generation.
* **initial\_time** (`float`, *optional*, defaults to `time.time()`) —
  The start of the generation allowed time.

This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
time will start being counted when you initialize this function. You can override this by passing an
`initial_time`.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L105)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional stopping criteria specific kwargs.

Returns

`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

### class transformers.StopStringCriteria

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L111)

( tokenizer: PreTrainedTokenizerBase stop\_strings: typing.Union[str, list[str]]  )

Parameters

* **tokenizer** (`PreTrainedTokenizer`) —
  The model’s associated tokenizer (necessary to extract vocab and tokenize the termination sequences)
* **stop\_strings** (`Union[str, list[str]]`) —
  A list of strings that should end generation. If a string is passed, it will be treated like a
  list with a single element.

This class can be used to stop generation whenever specific string sequences are generated. It preprocesses
the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

Generation is stopped as soon as a token is generated that completes any of the stop strings.
We want to catch any instance in which the stop string would be present in the decoded output, which means
we must also catch cases with “overhangs” off one or both ends. To make this more concrete, for the stop string
“stop”, any of the following token sequences would trigger the match:

* [“st”, “op”]
* [“stop”]
* [“st”, “opera”]
* [“sto”, “pper”]
* [“las”, “topper”]
* [“s”, “to”, “pped”]

Note that a match will only be triggered if the stop string is at the end of the generated sequence. In other
words, these sequences will not trigger a match:

* [“stop”, “at”]
* [“st”, “op”, “at”]
* [“st”, “opera”, “tion”]

The reason these are not a match is that the stop string does not overlap with the final token. If you can remove
one or more tokens from the end of the sequence without destroying the stop string, then this criterion will not
match that stop string. This is by design; because this check is run after each token is generated, we can’t miss a
valid stop string if one is generated, but we don’t want to halt generation just because the stop string exists
somewhere in the past input\_ids.

How is the match actually performed, though? We do it in quite a confusing way, because we want the entire match
process to be compilable with Torch or XLA, which means we cannot use standard string methods. However, it is possible,
with some work, to do string matching with pure tensor operations. We’ll begin by describing the algorithm we use
with standard string operations, and then at the end we’ll explain how this is converted to pure tensor operations.

The key to the algorithm is an observation: Because the stop string must overlap with the end of the token sequence, we can start at
the end of the sequence and work backwards. Specifically, we check that there is an overlap between the start of
the final token and the end of the stop\_string, or to put it another way, stop\_string[-i:] == token[:i] for
some i > 0. If you look at the positive examples above, you’ll see the last token in all of them fulfills this
property:

* [“st”, “op”] (overlap is “op”, overlap length == 2)
* [“stop”] (overlap is “stop”, overlap length == 4)
* [“st”, “opera”] (overlap is “op”, overlap length == 2)
* [“sto”, “pper”] (overlap is “p”, overlap length == 1)
* [“las”, “topper”] (overlap is “top”, overlap length == 3)
* [“s”, “to”, “pped”] (overlap is “p”, overlap length == 1)

It’s impossible to construct a matching sequence that does not have this property (feel free to verify this
yourself). However, although this overlap between the start of the final token and the end of the stop string is
necessary for a match, it is not sufficient. We also need to check that the rest of the token sequence is
consistent with the stop string.

How do we do that? Let’s use [“s”, “to”, “pped”] as an example. We know that the final token, “pped”, has an
overlap of 1 with the stop string, “stop”. We then go back to the previous token, “to”. Since we have already
matched 1 character from the stop string, the remainder to check is “sto”. We check that the next token “to”
matches the end of the remainder, which it does. We have now matched 3 characters from the stop string, and the
remainder to match is “s”. We go back to the previous token again, which is also “s”. This is a match, and so
we have matched the entire stop string.

How does it work when the tokens run off the start of the stop string, though? Let’s consider the example of
[“las”, “topper”]. The final token, “topper”, has an overlap of 3 with the stop string, “stop”. Therefore,
the remaining stop string to match is “s”. We go back to the previous token, “las”. Because the remainder to
match is just “s”, with length 1, we consider only the final 1 character from the token, which is “s”. This
matches the stop string, and so the entire string is matched.

How do we compute these matches with tensor operations, though? Simply: we efficiently precompute the necessary
information for all tokens! For every token, we compute:

* Its overlap with the end of the stop string, if any
* The positions inside the stop string where the token matches, including matches that run off the start.
* The total length of the token

For example, for the token “pped”, we would compute an end overlap of 1, no internal matching positions,
and a length of 4. For the token “to”, we would compute no end overlap, a single internal matching position
of 1 (counting from the end), and a length of 2. For the token “s”, we would compute no end overlap,
a single internal matching position of 3 (again counting from the end) and a length of 1.

As long as we have this information, we can execute the algorithm above without any string comparison
operations. We simply perform the following steps:

* Check if the final token has an end-overlap with the start string
* Continue backwards, keeping track of how much of the stop string we’ve matched so far
* At each point, check if the next token has the current position as one of its valid positions
* Continue until either a match fails, or we completely match the whole stop string

Again, consider [“s”, “to”, “pped”] as an example. “pped” has an end overlap of 1, so we can begin a match.
We have matched 1 character so far, so we check that the next token “to”, has 1 as a valid position (again,
counting from the end). It does, so we add the length of “to” to our position tracker. We have now matched
3 characters, so we check that the next token “s” has 3 as a valid position. It does, so we add its length
to the position tracker. The position tracker is now 4, which is the length of the stop string. We have matched the
entire stop string.

In the second case, [“las”, “topper”], “topper” has an end overlap of 3, so we can begin a match. We have
matched 3 characters so far, so we check that the next token “las” has 3 as a valid position. It does, because we
allow tokens to match positions that run off the start of the stop string. We add its length to the position
tracker. The position tracker is now 6, which is greater than the length of the stop string! Don’t panic, though -
this also counts as a match of the stop string. We have matched the entire stop string.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
>>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
>>> inputs = tokenizer("The biggest states in the USA by land area:", return_tensors="pt")

>>> gen_out = model.generate(**inputs)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
The biggest states in the USA by land area:
- Alaska
- Texas
- California

>>> # Passing one or more stop strings will halt generation after those strings are emitted
>>> # Note that generating with stop strings requires you to pass the tokenizer too
>>> gen_out = model.generate(**inputs, stop_strings=["Texas"], tokenizer=tokenizer)
>>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
The biggest states in the USA by land area:
- Alaska
- Texas
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L390)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional stopping criteria specific kwargs.

Returns

`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

### class transformers.EosTokenCriteria

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L452)

( eos\_token\_id: typing.Union[int, list[int], torch.Tensor]  )

Parameters

* **eos\_token\_id** (`Union[int, list[int], torch.Tensor]`) —
  The id(s) of the *end-of-sequence* token.

This class can be used to stop generation whenever the “end-of-sequence” token is generated.
By default, it uses the `model.generation_config.eos_token_id`.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/stopping_criteria.py#L469)

( input\_ids: LongTensor scores: FloatTensor \*\*kwargs  ) → `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) —
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional stopping criteria specific kwargs.

Returns

`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

## Constraints

A [Constraint](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Constraint) can be used to force the generation to include specific tokens or sequences in the output. Please note that this is exclusively available to our PyTorch implementations.

### class transformers.Constraint

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L5)

( )

Abstract base class for all constraints that can be applied during generation.
It must define how the constraint can be satisfied.

All classes that inherit Constraint must follow the requirement that


```
completed = False
while not completed:
    _, completed = constraint.update(constraint.advance())
```

will always terminate (halt).

#### advance

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L48)

( ) → token\_ids (Union[int, list[int], None])

Returns

token\_ids (Union[int, list[int], None])

* A single token ID (int) that advances the constraint, or
* A list of token IDs that could advance the constraint
* None if the constraint is completed or cannot be advanced

When called, returns the token(s) that would take this constraint one step closer to being fulfilled.

#### copy

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L116)

( stateful = False  ) → constraint(`Constraint`)

Parameters

* **stateful(`bool`)** — Whether to not only copy the constraint for new instance, but also its state.

Returns

constraint(`Constraint`)

The same constraint as the one being called from.

Creates a new instance of this constraint.

#### does\_advance

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L63)

( token\_id: int  )

Reads in a token and returns whether it creates progress.

#### remaining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L107)

( )

Returns the number of remaining steps of `advance()` in order to complete this constraint.

#### reset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L97)

( )

Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
a constraint is abrupted by an unwanted token.

#### test

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L24)

( )

Tests whether this constraint has been properly defined.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L72)

( token\_id: int  ) → stepped(`bool`)

Parameters

* **token\_id(`int`)** —
  The id of a newly generated token in the beam search.

Returns

stepped(`bool`)

Whether this constraint has become one step closer to being fulfuilled.
completed(`bool`):
Whether this constraint has been completely fulfilled by this token being generated.
reset (`bool`):
Whether this constraint has reset its progress by this token being generated.

Reads in a token and returns booleans that indicate the progress made by it. This function will update the
state of this object unlikes `does_advance(self, token_id: int)`.

This isn’t to test whether a certain token will advance the progress; it’s to update its state as if it has
been generated. This becomes important if token\_id != desired token (refer to else statement in
PhrasalConstraint)

### class transformers.PhrasalConstraint

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L132)

( token\_ids: list  )

Parameters

* **token\_ids** (`list[int]`) —
  The id of the token that must be generated by the output.

[Constraint](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Constraint) enforcing that an ordered sequence of tokens is included in the output.

### class transformers.DisjunctiveConstraint

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L264)

( nested\_token\_ids: list  )

Parameters

* **nested\_token\_ids** (`list[list[int]]`) —
  A list of words, where each word is a list of ids. This constraint is fulfilled by generating just one from
  the list of words.

A special [Constraint](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Constraint) that is fulfilled by fulfilling just one of several constraints.

### class transformers.ConstraintListState

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L354)

( constraints: list  )

Parameters

* **constraints** (`list[Constraint]`) —
  A list of [Constraint](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Constraint) objects that must be fulfilled by the beam scorer.

A class for beam scorers to track its progress through a list of constraints.

#### advance

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L386)

( )

The list of tokens to generate such that we can make progress.
By “list” we don’t mean the list of token that will fully fulfill a constraint.

Given constraints `c_i = {t_ij | j == # of tokens}`, If we’re not in the middle of progressing through a
specific constraint `c_i`, we return:

`[t_k1 for k in indices of unfulfilled constraints]`

If we are in the middle of a constraint, then we return:
`[t_ij]`, where `i` is the index of the inprogress constraint, `j` is the next step for the constraint.

Though we don’t care which constraint is fulfilled first, if we are in the progress of fulfilling a constraint,
that’s the only one we’ll return.

#### reset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L421)

( token\_ids: typing.Optional[list[int]]  )

token\_ids: the tokens generated thus far to reset the state of the progress through constraints.

## BeamSearch

### class transformers.BeamScorer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L91)

( )

Abstract base class for all beam scorers that are used for `~PreTrainedModel.beam_search` and
`~PreTrainedModel.beam_sample`.

#### process

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L97)

( input\_ids: LongTensor next\_scores: FloatTensor next\_tokens: LongTensor next\_indices: LongTensor \*\*kwargs  ) → `UserDict`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using any class inheriting from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **next\_scores** (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`) —
  Current scores of the top `2 * num_beams` non-finished beam hypotheses.
* **next\_tokens** (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`) —
  `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
* **next\_indices** (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`) —
  Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
* **pad\_token\_id** (`int`, *optional*) —
  The id of the *padding* token.
* **eos\_token\_id** (`Union[int, list[int]]`, *optional*) —
  The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
* **beam\_indices** (`torch.LongTensor`, *optional*) —
  Beam indices indicating to which beam hypothesis each token correspond.
* **group\_index** (`int`, *optional*) —
  The index of the group of beams. Used with `~PreTrainedModel.group_beam_search`.

Returns

`UserDict`

A dictionary composed of the fields as defined above:

* **next\_beam\_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) — Updated scores of all
  non-finished beams.
* **next\_beam\_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) — Next tokens to be added
  to the non-finished beam\_hypotheses.
* **next\_beam\_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) — Beam indices
  indicating to which beam the next tokens shall be added.

#### finalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L109)

( input\_ids: LongTensor next\_scores: FloatTensor next\_tokens: LongTensor next\_indices: LongTensor max\_length: int \*\*kwargs  ) → `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using any class inheriting from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **final\_beam\_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) —
  The final scores of all non-finished beams.
* **final\_beam\_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) —
  The last tokens to be added to the non-finished beam\_hypotheses.
* **final\_beam\_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) —
  The beam indices indicating to which beam the `final_beam_tokens` shall be added.
* **pad\_token\_id** (`int`, *optional*) —
  The id of the *padding* token.
* **eos\_token\_id** (`Union[int, list[int]]`, *optional*) —
  The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

Returns

`torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

The generated sequences.
The second dimension (sequence\_length) is either equal to `max_length` or shorter if all batches finished early
due to the `eos_token_id`.

### class transformers.BeamSearchScorer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L123)

( batch\_size: int num\_beams: int device: device length\_penalty: typing.Optional[float] = 1.0 do\_early\_stopping: typing.Union[bool, str, NoneType] = False num\_beam\_hyps\_to\_keep: typing.Optional[int] = 1 num\_beam\_groups: typing.Optional[int] = 1 max\_length: typing.Optional[int] = None  )

Parameters

* **batch\_size** (`int`) —
  Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
* **num\_beams** (`int`) —
  Number of beams for beam search.
* **device** (`torch.device`) —
  Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
  allocated.
* **length\_penalty** (`float`, *optional*, defaults to 1.0) —
  Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
  the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
  likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
  `length_penalty` < 0.0 encourages shorter sequences.
* **do\_early\_stopping** (`bool` or `str`, *optional*, defaults to `False`) —
  Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
  `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
  heuristic is applied and the generation stops when is it very unlikely to find better candidates;
  `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
  beam search algorithm).
* **num\_beam\_hyps\_to\_keep** (`int`, *optional*, defaults to 1) —
  The number of beam hypotheses that shall be returned upon calling
  [finalize()](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BeamSearchScorer.finalize).
* **num\_beam\_groups** (`int`, *optional*, defaults to 1) —
  Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
  See [this paper](https://huggingface.co/papers/1610.02424) for more details.
* **max\_length** (`int`, *optional*) —
  The maximum length of the sequence to be generated.

[BeamScorer](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BeamScorer) implementing standard beam search decoding.

Adapted in part from [Facebook’s XLM beam search
code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan’s DBS
implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

#### process

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L215)

( input\_ids: LongTensor next\_scores: FloatTensor next\_tokens: LongTensor next\_indices: LongTensor pad\_token\_id: typing.Union[int, torch.Tensor, NoneType] = None eos\_token\_id: typing.Union[int, list[int], torch.Tensor, NoneType] = None beam\_indices: typing.Optional[torch.LongTensor] = None group\_index: typing.Optional[int] = 0 decoder\_prompt\_len: typing.Optional[int] = 0  )

#### finalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L320)

( input\_ids: LongTensor final\_beam\_scores: FloatTensor final\_beam\_tokens: LongTensor final\_beam\_indices: LongTensor max\_length: int pad\_token\_id: typing.Union[int, torch.Tensor, NoneType] = None eos\_token\_id: typing.Union[int, list[int], torch.Tensor, NoneType] = None beam\_indices: typing.Optional[torch.LongTensor] = None decoder\_prompt\_len: typing.Optional[int] = 0  )

### class transformers.ConstrainedBeamSearchScorer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L419)

( batch\_size: int num\_beams: int constraints: list device: device length\_penalty: typing.Optional[float] = 1.0 do\_early\_stopping: typing.Union[bool, str, NoneType] = False num\_beam\_hyps\_to\_keep: typing.Optional[int] = 1 num\_beam\_groups: typing.Optional[int] = 1 max\_length: typing.Optional[int] = None  )

Parameters

* **batch\_size** (`int`) —
  Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
* **num\_beams** (`int`) —
  Number of beams for beam search.
* **constraints** (`list[Constraint]`) —
  A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
  output. For more information, the documentation of [Constraint](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Constraint) should be read.
* **device** (`torch.device`) —
  Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
  allocated.
* **length\_penalty** (`float`, *optional*, defaults to 1.0) —
  Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
  the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
  likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
  `length_penalty` < 0.0 encourages shorter sequences.
* **do\_early\_stopping** (`bool` or `str`, *optional*, defaults to `False`) —
  Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
  `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
  heuristic is applied and the generation stops when is it very unlikely to find better candidates;
  `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
  beam search algorithm).
* **num\_beam\_hyps\_to\_keep** (`int`, *optional*, defaults to 1) —
  The number of beam hypotheses that shall be returned upon calling
  [finalize()](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BeamSearchScorer.finalize).
* **num\_beam\_groups** (`int`, *optional*, defaults to 1) —
  Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
  See [this paper](https://huggingface.co/papers/1610.02424) for more details.
* **max\_length** (`int`, *optional*) —
  The maximum length of the sequence to be generated.

[BeamScorer](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BeamScorer) implementing constrained beam search decoding.

#### process

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L513)

( input\_ids: LongTensor next\_scores: FloatTensor next\_tokens: LongTensor next\_indices: LongTensor scores\_for\_all\_vocab: FloatTensor pad\_token\_id: typing.Union[int, torch.Tensor, NoneType] = None eos\_token\_id: typing.Union[int, list[int], torch.Tensor, NoneType] = None beam\_indices: typing.Optional[torch.LongTensor] = None decoder\_prompt\_len: typing.Optional[int] = 0  ) → `UserDict`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using any class inheriting from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **next\_scores** (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`) —
  Current scores of the top `2 * num_beams` non-finished beam hypotheses.
* **next\_tokens** (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`) —
  `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
* **next\_indices** (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`) —
  Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
* **scores\_for\_all\_vocab** (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`) —
  The scores of all tokens in the vocabulary for each of the beam hypotheses.
* **pad\_token\_id** (`int`, *optional*) —
  The id of the *padding* token.
* **eos\_token\_id** (`Union[int, list[int]]`, *optional*) —
  The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
* **beam\_indices** (`torch.LongTensor`, *optional*) —
  Beam indices indicating to which beam hypothesis each token correspond.
* **decoder\_prompt\_len** (`int`, *optional*) —
  The length of prompt that is included in the input to decoder.

Returns

`UserDict`

A dictionary composed of the fields as defined above:

* **next\_beam\_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) — Updated scores of
  all
  non-finished beams.
* **next\_beam\_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) — Next tokens to be
  added
  to the non-finished beam\_hypotheses.
* **next\_beam\_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) — Beam indices
  indicating to which beam the next tokens shall be added.

#### finalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L813)

( input\_ids: LongTensor final\_beam\_scores: FloatTensor final\_beam\_tokens: LongTensor final\_beam\_indices: LongTensor max\_length: int pad\_token\_id: typing.Union[int, torch.Tensor, NoneType] = None eos\_token\_id: typing.Union[int, list[int], torch.Tensor, NoneType] = None beam\_indices: typing.Optional[torch.LongTensor] = None decoder\_prompt\_len: typing.Optional[int] = 0  )

## Streamers

### class transformers.TextStreamer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L41)

( tokenizer: AutoTokenizer skip\_prompt: bool = False \*\*decode\_kwargs  )

Parameters

* **tokenizer** (`AutoTokenizer`) —
  The tokenized used to decode the tokens.
* **skip\_prompt** (`bool`, *optional*, defaults to `False`) —
  Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
* **decode\_kwargs** (`dict`, *optional*) —
  Additional keyword arguments to pass to the tokenizer’s `decode` method.

Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

The API for the streamer classes is still under development and may change in the future.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

#### end

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L119)

( )

Flushes any remaining cache and prints a newline to stdout.

#### on\_finalized\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L133)

( text: str stream\_end: bool = False  )

Prints the new text to stdout. If the stream is ending, also prints a newline.

#### put

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L85)

( value  )

Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.

### class transformers.TextIteratorStreamer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L162)

( tokenizer: AutoTokenizer skip\_prompt: bool = False timeout: float | None = None \*\*decode\_kwargs  )

Parameters

* **tokenizer** (`AutoTokenizer`) —
  The tokenized used to decode the tokens.
* **skip\_prompt** (`bool`, *optional*, defaults to `False`) —
  Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
* **timeout** (`float`, *optional*) —
  The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
  in `.generate()`, when it is called in a separate thread.
* **decode\_kwargs** (`dict`, *optional*) —
  Additional keyword arguments to pass to the tokenizer’s `decode` method.

Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
useful for applications that benefit from accessing the generated text in a non-blocking way (e.g. in an interactive
Gradio demo).

The API for the streamer classes is still under development and may change in the future.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
>>> from threading import Thread

>>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextIteratorStreamer(tok)

>>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
>>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
>>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
>>> thread.start()
>>> generated_text = ""
>>> for new_text in streamer:
...     generated_text += new_text
>>> generated_text
'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
```

#### on\_finalized\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L216)

( text: str stream\_end: bool = False  )

Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.

### class transformers.AsyncTextIteratorStreamer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L233)

( tokenizer: AutoTokenizer skip\_prompt: bool = False timeout: float | None = None \*\*decode\_kwargs  )

Parameters

* **tokenizer** (`AutoTokenizer`) —
  The tokenized used to decode the tokens.
* **skip\_prompt** (`bool`, *optional*, defaults to `False`) —
  Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
* **timeout** (`float`, *optional*) —
  The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
  in `.generate()`, when it is called in a separate thread.
* **decode\_kwargs** (`dict`, *optional*) —
  Additional keyword arguments to pass to the tokenizer’s `decode` method.

Raises

`TimeoutError`

* `TimeoutError` — If token generation time exceeds timeout value.

Streamer that stores print-ready text in a queue, to be used by a downstream application as an async iterator.
This is useful for applications that benefit from accessing the generated text asynchronously (e.g. in an
interactive Gradio demo).

The API for the streamer classes is still under development and may change in the future.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, AsyncTextIteratorStreamer
>>> from threading import Thread
>>> import asyncio

>>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")

>>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
>>> async def main():
...     # Important: AsyncTextIteratorStreamer must be initialized inside a coroutine!
...     streamer = AsyncTextIteratorStreamer(tok)
...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
...     thread.start()
...     generated_text = ""
...     async for new_text in streamer:
...         generated_text += new_text
>>>     print(generated_text)
>>> asyncio.run(main())
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

#### on\_finalized\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/streamers.py#L296)

( text: str stream\_end: bool = False  )

Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.

## Caches

### class transformers.CacheLayerMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L26)

( )

Base, abstract class for a single layer’s cache.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L40)

( key\_states: Tensor value\_states: Tensor cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  )

#### get\_seq\_length

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L48)

( )

#### get\_mask\_sizes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L45)

( cache\_position: Tensor  )

#### get\_max\_cache\_shape

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L51)

( )

#### reset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L66)

( )

Resets the cache values while preserving the objects

#### reorder\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L75)

( beam\_idx: LongTensor  )

Reorders this layer’s cache for beam search.

#### lazy\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L37)

( key\_states: Tensor  )

### class transformers.DynamicLayer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L82)

( )

A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L95)

( key\_states: Tensor value\_states: Tensor cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  ) → tuple[`torch.Tensor`, `torch.Tensor`]

Parameters

* **key\_states** (`torch.Tensor`) — The new key states to cache.
* **value\_states** (`torch.Tensor`) — The new value states to cache.
* **cache\_kwargs** (`dict[str, Any]`, *optional*) — Additional arguments for the cache.

Returns

tuple[`torch.Tensor`, `torch.Tensor`]

The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

#### lazy\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L90)

( key\_states: Tensor  )

#### crop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L138)

( max\_length: int  )

Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
to remove `max_length` tokens.

#### batch\_repeat\_interleave

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L152)

( repeats: int  )

Repeat the cache `repeats` times in the batch dimension.

#### batch\_select\_indices

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L158)

( indices: Tensor  )

Only keep the `indices` in the batch dimension of the cache.

### class transformers.StaticLayer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L247)

( max\_cache\_len: int  )

Parameters

* **max\_cache\_len** (`int`) —
  Maximum number of tokens that can be stored, used for tensor preallocation.

A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L299)

( key\_states: Tensor value\_states: Tensor cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  ) → tuple[`torch.Tensor`, `torch.Tensor`]

Parameters

* **key\_states** (`torch.Tensor`) — The new key states to cache.
* **value\_states** (`torch.Tensor`) — The new value states to cache.
* **cache\_kwargs** (`dict[str, Any]`, *optional*) — Additional arguments for the cache.

Returns

tuple[`torch.Tensor`, `torch.Tensor`]

The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

#### lazy\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L264)

( key\_states: Tensor  )

Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
num\_heads in case of TP etc…) at runtime directly, which is extremely practical as it avoids moving
devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
internally don’t compile the prefill, this is guaranteed to have been called already when compiling.
If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
not be compiled anyway for performances!

### class transformers.SlidingWindowLayer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L354)

( max\_cache\_len: int sliding\_window: int  )

Parameters

* **max\_cache\_len** (`int`) —
  Maximum number of tokens that can be stored, used for tensor preallocation.
* **sliding\_window** (`int`) —
  The size of the sliding window.

A static cache layer that stores the key and value states as static tensors of shape
`[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing
tensors, and then mutates them in-place. Built for `torch.compile` support.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L374)

( key\_states: Tensor value\_states: Tensor cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  ) → tuple[`torch.Tensor`, `torch.Tensor`]

Parameters

* **key\_states** (`torch.Tensor`) — The new key states to cache.
* **value\_states** (`torch.Tensor`) — The new value states to cache.
* **cache\_kwargs** (`dict[str, Any]`, *optional*) — Additional arguments for the cache.

Returns

tuple[`torch.Tensor`, `torch.Tensor`]

The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

#### lazy\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L264)

( key\_states: Tensor  )

Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
num\_heads in case of TP etc…) at runtime directly, which is extremely practical as it avoids moving
devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
internally don’t compile the prefill, this is guaranteed to have been called already when compiling.
If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
not be compiled anyway for performances!

### class transformers.QuantoQuantizedLayer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L614)

( nbits: int = 4 axis\_key: int = 0 axis\_value: int = 0 q\_group\_size: int = 64 residual\_length: int = 128  )

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L562)

( key\_states: Tensor value\_states: Tensor cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  ) → tuple[`torch.Tensor`, `torch.Tensor`]

Parameters

* **key\_states** (`torch.Tensor`) — The new key states to cache.
* **value\_states** (`torch.Tensor`) — The new value states to cache.
* **cache\_kwargs** (`dict[str, Any]`, *optional*) — Additional arguments for the cache.

Returns

tuple[`torch.Tensor`, `torch.Tensor`]

The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

#### lazy\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L90)

( key\_states: Tensor  )

### class transformers.HQQQuantizedLayer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L664)

( nbits: int = 4 axis\_key: int = 0 axis\_value: int = 0 q\_group\_size: int = 64 residual\_length: int = 128  )

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L562)

( key\_states: Tensor value\_states: Tensor cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  ) → tuple[`torch.Tensor`, `torch.Tensor`]

Parameters

* **key\_states** (`torch.Tensor`) — The new key states to cache.
* **value\_states** (`torch.Tensor`) — The new value states to cache.
* **cache\_kwargs** (`dict[str, Any]`, *optional*) — Additional arguments for the cache.

Returns

tuple[`torch.Tensor`, `torch.Tensor`]

The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

#### lazy\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L90)

( key\_states: Tensor  )

### class transformers.Cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L718)

( layers: typing.Optional[list[transformers.cache\_utils.CacheLayerMixin]] = None layer\_class\_to\_replicate: typing.Optional[type[transformers.cache\_utils.CacheLayerMixin]] = None offloading: bool = False offload\_only\_non\_sliding: bool = True  )

Parameters

* **layers** (`Optional`, *optional*) —
  A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
  be used.
* **layer\_class\_to\_replicate** (`type[CacheLayerMixin]`, *optional*) —
  Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
  and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
  list of layers.
* **offloading** (`bool`, *optional*, defaults to `False`) —
  Whether to perform offloading of the layers to `cpu`, to save GPU memory.
* **offload\_only\_non\_sliding** (`bool`, *optional*, defaults to `True`) —
  If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
  usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
the Cache of each layer.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L794)

( key\_states: Tensor value\_states: Tensor layer\_idx: int cache\_kwargs: typing.Optional[dict[str, typing.Any]] = None  )

Parameters

* **key\_states** (`torch.Tensor`) —
  The new key states to cache.
* **value\_states** (`torch.Tensor`) —
  The new value states to cache.
* **layer\_idx** (`int`) —
  The index of the layer to cache the states for.
* **cache\_kwargs** (`dict[str, Any]`, *optional*) —
  Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
  cache to be created.

Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#### early\_initialization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L835)

( batch\_size: int num\_heads: int head\_dim: int dtype: dtype device: device  )

Initialize all the layers in advance (it’s otherwise lazily initialized on the first `update` call).
This is useful for our `export` recipes, as `export` needs everything in advance.

#### get\_seq\_length

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L850)

( layer\_idx: typing.Optional[int] = 0  )

Returns the sequence length of the cache for the given layer.

#### get\_mask\_sizes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L856)

( cache\_position: Tensor layer\_idx: int  )

Return a tuple (kv\_length, kv\_offset) corresponding to the length and offset that will be returned for
the given layer at `layer_idx`.
The masks are then prepared according to the given lengths (kv\_length, kv\_offset) and patterns for each layer.

#### get\_max\_cache\_shape

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L868)

( layer\_idx: int = 0  )

Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length.

#### reset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L876)

( )

Recursively reset all layers tensors

#### reorder\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L881)

( beam\_idx: LongTensor  )

Reorder the cache for beam search

#### crop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L886)

( max\_length: int  )

Crop the cache to the given length

#### batch\_repeat\_interleave

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L891)

( repeats: int  )

Repeat and interleave the cache

#### batch\_select\_indices

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L896)

( indices: Tensor  )

Select indices from the cache

### class transformers.DynamicCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L959)

( ddp\_cache\_data: typing.Optional[collections.abc.Iterable[tuple[torch.Tensor, torch.Tensor]]] = None config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None offloading: bool = False offload\_only\_non\_sliding: bool = False  )

Parameters

* **ddp\_cache\_data** (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*) —
  It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
  `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
  for a layer gathered across replicas by torch.distributed (shape=[global batch size, num\_heads, seq\_len, head\_dim]).
  Note: it needs to be the 1st arg as well to work correctly
* **config** (`PretrainedConfig`, *optional*) —
  The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
  or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
  `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
* **offloading** (`bool`, *optional*, defaults to `False`) —
  Whether to perform offloading of the layers to `cpu`, to save GPU memory.
* **offload\_only\_non\_sliding** (`bool`, *optional*, defaults to `False`) —
  If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
  usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

A cache that grows dynamically as more tokens are generated. This is the default for generative models.
It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

See `Cache` for details on common methods that are implemented by all cache classes.

Example:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

>>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

>>> # Prepare a cache class and pass it to model's forward
>>> past_key_values = DynamicCache(config=model.config)
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values # access cache filled with key/values from generation
```

#### to\_legacy\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1050)

( )

Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
backward compatibility.

#### from\_legacy\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1060)

( past\_key\_values: tuple  )

Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
backward compatibility.

### class transformers.QuantizedCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1151)

( backend: str config: PretrainedConfig nbits: int = 4 axis\_key: int = 0 axis\_value: int = 0 q\_group\_size: int = 64 residual\_length: int = 128  )

Parameters

* **backend** (`str`) —
  The quantization backend to use. One of `(“quanto”, “hqq”).
* **config** (`PretrainedConfig`) —
  The config of the model for which this Cache will be used.
* **nbits** (`int`, *optional*, defaults to 4) —
  The number of bits for quantization.
* **axis\_key** (`int`, *optional*, defaults to 0) —
  The axis on which to quantize the keys.
* **axis\_value** (`int`, *optional*, defaults to 0) —
  The axis on which to quantize the values.
* **q\_group\_size** (`int`, *optional*, defaults to 64) —
  Quantization is done per-channel according to a set `q_group_size` for both keys and values.
* **residual\_length** (`int`, *optional*, defaults to 128) —
  Maximum capacity for the original precision cache

A quantizer cache similar to what is described in the
[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
It allows the model to generate longer sequence length without allocating too much memory for keys and values
by applying quantization.
The cache has two types of storage, one for original precision and one for the
quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
described in the paper.

See `Cache` for details on common methods that are implemented by all cache classes.

### class transformers.QuantoQuantizedCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1470)

( config: PretrainedConfig nbits: int = 4 axis\_key: int = 0 axis\_value: int = 0 q\_group\_size: int = 64 residual\_length: int = 128  )

### class transformers.HQQQuantizedCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1487)

( config: PretrainedConfig nbits: int = 4 axis\_key: int = 0 axis\_value: int = 0 q\_group\_size: int = 64 residual\_length: int = 128  )

### class transformers.OffloadedCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1416)

( )

### class transformers.StaticCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1076)

( config: PretrainedConfig max\_cache\_len: int offloading: bool = False offload\_only\_non\_sliding: bool = True \*\*kwargs  )

Parameters

* **config** (`PretrainedConfig`) —
  The config of the model for which this Cache will be used. It will be used to check for sliding
  or hybrid layer structure, and initialize each layer accordingly.
* **max\_cache\_len** (`int`) —
  The maximum number of tokens that this Cache should hold.
* **offloading** (`bool`, *optional*, defaults to `False`) —
  Whether to perform offloading of the layers to `cpu`, to save GPU memory.
* **offload\_only\_non\_sliding** (`bool`, *optional*, defaults to `True`) —
  If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
  usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

Static Cache class to be used with `torch.compile(model)` and `torch.export()`. It will check the `config`
for potential hybrid cache structure, and initialize each layer accordingly.

See `Cache` for details on common methods that are implemented by all cache classes.

Example:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

>>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

>>> # Prepare a cache class and pass it to model's forward
>>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
>>> max_generated_length = inputs.input_ids.shape[1] + 10
>>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values # access cache filled with key/values from generation
StaticCache()
```

### class transformers.OffloadedStaticCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1425)

( config: PretrainedConfig max\_cache\_len: int \*args \*\*kwargs  )

### class transformers.HybridCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1443)

( config: PretrainedConfig max\_cache\_len: int \*args \*\*kwargs  )

### class transformers.HybridChunkedCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1452)

( config: PretrainedConfig max\_cache\_len: int \*args \*\*kwargs  )

### class transformers.SlidingWindowCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1434)

( config: PretrainedConfig max\_cache\_len: int \*args \*\*kwargs  )

### class transformers.EncoderDecoderCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1207)

( \*caches  )

Parameters

* **caches** (`Iterable`) —
  Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the
  second one for cross-attention. Can optionally also be an iterable of length 1, containing a
  `tuple[tuple[torch.Tensor]]` (usually used for compatibility with torch dp and ddp).

Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
cross-attention caches.

See `Cache` for details on common methods that are implemented by all cache classes.

Example:


```
>>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

>>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
>>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

>>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

>>> # Prepare cache classes for encoder and decoder and pass it to model's forward
>>> self_attention_cache = DynamicCache(config=self.config)
>>> cross_attention_cache = DynamicCache(config=self.config)
>>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values # access cache filled with key/values from generation
EncoderDecoderCache()
```

#### to\_legacy\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1307)

( )

Converts the `EncoderDecoderCache` instance into its equivalent in the legacy cache format.

#### from\_legacy\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py#L1319)

( past\_key\_values: typing.Optional[collections.abc.Iterable[tuple[torch.FloatTensor, ...]]]  )

Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`.

## Watermark Utils

### class transformers.WatermarkingConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L1331)

( greenlist\_ratio: typing.Optional[float] = 0.25 bias: typing.Optional[float] = 2.0 hashing\_key: typing.Optional[int] = 15485863 seeding\_scheme: typing.Optional[str] = 'lefthash' context\_width: typing.Optional[int] = 1  )

Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
See [this paper](https://huggingface.co/papers/2306.04634) for more details on the arguments.

Accepts the following keys:

* greenlist\_ratio (`float`):
  Used for watermarking. The ratio of “green” tokens used to the vocabulary size. Defaults to 0.25.
* bias (`float`):
  Used with watermarking. The bias added to the selected “green” tokens’ logits. Defaults to 2.0.
* hashing\_key (`int`):
  Hashing key used for watermarking. Defaults to 15485863 (the millionth prime).
* seeding\_scheme (`str`):
  Algorithm to use for watermarking. Accepts values:
  + “lefthash” (default): “green” tokens selection depend on the last token (Algorithm 2 from the paper)
  + “selfhash”: “green” tokens selection depends on the current token itself (Algorithm 3 from the paper)
    The downside of this scheme is that it considers all possible next tokens and can be slower than “lefthash”.
* context\_width(`int`):
  The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.

#### \_\_call\_\_

 

( \*args \*\*kwargs  )

Call self as a function.

### class transformers.WatermarkDetector

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L73)

( model\_config: PretrainedConfig device: str watermarking\_config: typing.Union[transformers.generation.configuration\_utils.WatermarkingConfig, dict] ignore\_repeated\_ngrams: bool = False max\_cache\_size: int = 128  )

Parameters

* **model\_config** (`PretrainedConfig`) —
  The model config that will be used to get model specific arguments used when generating.
* **device** (`str`) —
  The device which was used during watermarked text generation.
* **watermarking\_config** (Union[`WatermarkingConfig`, `Dict`]) —
  The exact same watermarking config and arguments used when generating text.
* **ignore\_repeated\_ngrams** (`bool`, *optional*, defaults to `False`) —
  Whether to count every unique ngram only once or not.
* **max\_cache\_size** (`int`, *optional*, defaults to 128) —
  The max size to be used for LRU caching of seeding/sampling algorithms called for every token.

Detector for detection of watermark generated text. The detector needs to be given the exact same settings that were
given during text generation to replicate the watermark greenlist generation and so detect the watermark. This includes
the correct device that was used during text generation, the correct watermarking arguments and the correct tokenizer vocab size.
The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

See [the paper](https://huggingface.co/papers/2306.04634) for more information.

Examples:


```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

>>> model_id = "openai-community/gpt2"
>>> model = AutoModelForCausalLM.from_pretrained(model_id)
>>> tok = AutoTokenizer.from_pretrained(model_id)
>>> tok.pad_token_id = tok.eos_token_id
>>> tok.padding_side = "left"

>>> inputs = tok(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
>>> input_len = inputs["input_ids"].shape[-1]

>>> # first generate text with watermark and without
>>> watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
>>> out_watermarked = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=20)
>>> out = model.generate(**inputs, do_sample=False, max_length=20)

>>> # now we can instantiate the detector and check the generated text
>>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
>>> detection_out_watermarked = detector(out_watermarked, return_dict=True)
>>> detection_out = detector(out, return_dict=True)
>>> detection_out_watermarked.prediction
array([ True,  True])

>>> detection_out.prediction
array([False,  False])
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L193)

( input\_ids: LongTensor z\_threshold: float = 3.0 return\_dict: bool = False  ) → `WatermarkDetectorOutput` or `np.array`

Parameters

* **input\_ids** (`torch.LongTensor`) —
  The watermark generated text. It is advised to remove the prompt, which can affect the detection.
* **z\_threshold** (`Dict`, *optional*, defaults to `3.0`) —
  Changing this threshold will change the sensitivity of the detector. Higher z threshold gives less
  sensitivity and vice versa for lower z threshold.
* **return\_dict** (`bool`, *optional*, defaults to `False`) —
  Whether to return `~generation.WatermarkDetectorOutput` or not. If not it will return boolean predictions,

Returns

`WatermarkDetectorOutput` or `np.array`

A `WatermarkDetectorOutput`
if `return_dict=True` otherwise a `np.array`.

ma

### class transformers.BayesianDetectorConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L245)

( watermarking\_depth: typing.Optional[int] = None base\_rate: float = 0.5 \*\*kwargs  )

Parameters

* **watermarking\_depth** (`int`, *optional*) —
  The number of tournament layers.
* **base\_rate** (`float1`, *optional*, defaults to 0.5) —
  Prior probability P(w) that a text is watermarked.

This is the configuration class to store the configuration of a [BayesianDetectorModel](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BayesianDetectorModel). It is used to
instantiate a Bayesian Detector model according to the specified arguments.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

### class transformers.BayesianDetectorModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L352)

( config  )

Parameters

* **config** ([BayesianDetectorConfig](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BayesianDetectorConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bayesian classifier for watermark detection.

This detector uses Bayes’ rule to compute a watermarking score, which is the sigmoid of the log of ratio of the
posterior probabilities P(watermarked|g\_values) and P(unwatermarked|g\_values). Please see the section on
BayesianScore in the paper for further details.
Paper URL: <https://www.nature.com/articles/s41586-024-08025-4>

Note that this detector only works with non-distortionary Tournament-based watermarking using the Bernoulli(0.5)
g-value distribution.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L438)

( g\_values: Tensor mask: Tensor labels: typing.Optional[torch.Tensor] = None loss\_batch\_weight = 1 return\_dict = False  )

Parameters

* **g\_values** (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth, ...)`) —
  g-values (with values 0 or 1)
* **mask** —
  A binary array shape [batch\_size, seq\_len] indicating which g-values should be used. g-values with mask
  value 0 are discarded.

Computes the watermarked posterior P(watermarked|g\_values).

### class transformers.SynthIDTextWatermarkingConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L1409)

( ngram\_len: int keys: list context\_history\_size: int = 1024 sampling\_table\_seed: int = 0 sampling\_table\_size: int = 65536 skip\_first\_ngram\_calls: bool = False debug\_mode: bool = False  )

Parameters

* **ngram\_len** (`int`) —
  Ngram length.
* **keys** (`list[int]`) —
  A sequence of watermarking keys, one for each depth.
* **context\_history\_size** (`int`, *optional*, defaults to 1024) —
  Size of the tensor to keep track of seen contexts.
* **sampling\_table\_seed** (`int`, *optional*, defaults to 0) —
  Random seed to generate the sampling table.
* **sampling\_table\_size** (`int`, *optional*, defaults to 65536) —
  Size of the sampling table.
* **skip\_first\_ngram\_calls** (`bool`, *optional*, defaults to `False`) —
  Whether to skip first ngram calls.
* **debug\_mode** (`bool`, optional, *optional*, defaults to `False`) —
  Logits are modified to uniform one got before watermarking modification is applied. This is to test the
  implementation.

Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
See [this paper](https://www.nature.com/articles/s41586-024-08025-4) for more details on the arguments.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

>>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
>>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

>>> # SynthID Text configuration
>>> watermarking_config = SynthIDTextWatermarkingConfig(
...     keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
...     ngram_len=5,
... )

>>> # Generation with watermarking
>>> tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
>>> output_sequences = model.generate(
...     **tokenized_prompts, watermarking_config=watermarking_config, do_sample=True, max_new_tokens=10
... )
>>> watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
```

### class transformers.SynthIDTextWatermarkDetector

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L482)

( detector\_module: BayesianDetectorModel logits\_processor: SynthIDTextWatermarkLogitsProcessor tokenizer: typing.Any  )

Parameters

* **detector\_module** ([BayesianDetectorModel](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.BayesianDetectorModel)) —
  Bayesian detector module object initialized with parameters.
  Check <https://github.com/huggingface/transformers-research-projects/tree/main/synthid_text> for usage.
* **logits\_processor** (`SynthIDTextWatermarkLogitsProcessor`) —
  The logits processor used for watermarking.
* **tokenizer** (`Any`) —
  The tokenizer used for the model.

SynthID text watermark detector class.

This class has to be initialized with the trained bayesian detector module check script
in examples/synthid\_text/detector\_training.py for example in training/saving/loading this
detector module. The folder also showcases example use case of this detector.

Examples:


```
>>> from transformers import (
...     AutoTokenizer, BayesianDetectorModel, SynthIDTextWatermarkLogitsProcessor, SynthIDTextWatermarkDetector
... )

>>> # Load the detector. See https://github.com/huggingface/transformers-research-projects/tree/main/synthid_text for training a detector.
>>> detector_model = BayesianDetectorModel.from_pretrained("joaogante/dummy_synthid_detector")
>>> logits_processor = SynthIDTextWatermarkLogitsProcessor(
...     **detector_model.config.watermarking_config, device="cpu"
... )
>>> tokenizer = AutoTokenizer.from_pretrained(detector_model.config.model_name)
>>> detector = SynthIDTextWatermarkDetector(detector_model, logits_processor, tokenizer)

>>> # Test whether a certain string is watermarked
>>> test_input = tokenizer(["This is a test input"], return_tensors="pt")
>>> is_watermarked = detector(test_input.input_ids)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/watermarking.py#L529)

( tokenized\_outputs: Tensor  )

## Compile Utils

### class transformers.CompileConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L1499)

( fullgraph: bool = False dynamic: typing.Optional[bool] = None backend: typing.Union[str, typing.Callable] = 'inductor' mode: str = 'reduce-overhead' options: typing.Optional[dict] = None  )

Parameters

* **fullgraph** (`bool`, *optional*, defaults to `False`) —
  If False (default), attempts to discover compileable regions that will be optimized. If True, then require
  that the entire function be capturable into a single graph. If this is not possible (that is, if there are
  graph breaks), then an error will be raised.
* **dynamic** (`bool` or `None`, *optional*) —
  Whether to try to use dynamic shape graphs.
* **backend** (`str` or `Callable`, *optional*, defaults to `"inductor"`) —
  Backend to be used.
* **mode** (`str`, *optional*, defaults to `"reduce-overhead"`) —
  Controls balance between performance and overhead.
* **options** (`dict`, *optional*) —
  A dictionary of options to pass to the backend.

Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details on the arguments.

Examples:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

>>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
>>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b').cuda()

>>> # Automatic compile configuration, used with static cache
>>> compile_config = CompileConfig(dynamic=True)

>>> # Generation with static cache and compile config
>>> input = tokenizer.encode("Hello there, how", return_tensors="pt").cuda()
>>> output = model.generate(
...     input, do_sample=False, max_new_tokens=300, cache_implementation="static", compile_config=compile_config
... )
>>> output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

#### \_\_call\_\_

 

( \*args \*\*kwargs  )

Call self as a function.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/generation_utils.md)
