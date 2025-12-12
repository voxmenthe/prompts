# Utilities for Generation

This page lists all the utility functions used by [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate).

## Generate Outputs[[transformers.generation.GenerateDecoderOnlyOutput]]

The output of [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) is an instance of a subclass of
[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput). This output is a data structure containing all the information returned
by [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate), but that can also be used as tuple or dictionary.

Here's an example:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

The `generation_output` object is a [GenerateDecoderOnlyOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput), as we can
see in the documentation of that class below, it means it has the following attributes:

- `sequences`: the generated sequences of tokens
- `scores` (optional): the prediction scores of the language modelling head, for each generation step
- `hidden_states` (optional): the hidden states of the model, for each generation step
- `attentions` (optional): the attention weights of the model, for each generation step

Here we have the `scores` since we passed along `output_scores=True`, but we don't have `hidden_states` and
`attentions` because we didn't pass `output_hidden_states=True` or `output_attentions=True`.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get `None`. Here for instance `generation_output.scores` are all the generated prediction scores of the
language modeling head, and `generation_output.attentions` is `None`.

When using our `generation_output` object as a tuple, it only keeps the attributes that don't have `None` values.
Here, for instance, it has two elements, `loss` then `logits`, so

```python
generation_output[:2]
```

will return the tuple `(generation_output.sequences, generation_output.scores)` for instance.

When using our `generation_output` object as a dictionary, it only keeps the attributes that don't have `None`
values. Here, for instance, it has two keys that are `sequences` and `scores`.

We document here all output types.

#### transformers.generation.GenerateDecoderOnlyOutput[[transformers.generation.GenerateDecoderOnlyOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L150)

Outputs of decoder-only generation models, when using non-beam methods.

**Parameters:**

sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.

scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) : Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.

logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) : Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.

attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.

hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.

past_key_values (`Cache`, *optional*, returned when `use_cache=True`) : Returns the model cache, used to speed up decoding. Different models have a different cache format, check the model's documentation. Usually, a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance.

#### transformers.generation.GenerateEncoderDecoderOutput[[transformers.generation.GenerateEncoderDecoderOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L186)

Outputs of encoder-decoder generation models, when using non-beam methods.

**Parameters:**

sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`) : The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.

scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) : Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.

logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) : Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.

cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.

decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.

past_key_values (`Cache`, *optional*, returned when `use_cache=True`) : Returns the model cache, used to speed up decoding. Different models have a different cache format, check the model's documentation. Usually, a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance.

#### transformers.generation.GenerateBeamDecoderOnlyOutput[[transformers.generation.GenerateBeamDecoderOnlyOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L234)

Outputs of decoder-only generation models, when using beam methods.

**Parameters:**

sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`) : The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.

sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`) : Final beam scores of the generated `sequences`.

scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) : Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.

logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) : Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.

beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True`) : Beam indices of generated token id at each generation step. `torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`.

attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.

hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.

past_key_values (`Cache`, *optional*, returned when `use_cache=True`) : Returns the model cache, used to speed up decoding. Different models have a different cache format, check the model's documentation. Usually, a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance.

#### transformers.generation.GenerateBeamEncoderDecoderOutput[[transformers.generation.GenerateBeamEncoderDecoderOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L278)

Outputs of encoder-decoder generation models, when using beam methods.

**Parameters:**

sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`) : The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.

sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`) : Final beam scores of the generated `sequences`.

scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) : Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.

logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) : Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.

beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True`) : Beam indices of generated token id at each generation step. `torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.

decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length, sequence_length)`.

cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.

decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) : Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.

past_key_values (`Cache`, *optional*, returned when `use_cache=True`) : Returns the model cache, used to speed up decoding. Different models have a different cache format, check the model's documentation. Usually, a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance.

## LogitsProcessor[[transformers.AlternatingCodebooksLogitsProcessor]]

A [LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) can be used to modify the prediction scores of a language model head for
generation.

#### transformers.AlternatingCodebooksLogitsProcessor[[transformers.AlternatingCodebooksLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2298)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) enforcing alternated generation between the two codebooks of Bark.

This logits processor is exclusively compatible with
[Bark](https://huggingface.co/docs/transformers/en/model_doc/bark)'s fine submodel. See the model documentation
for examples.

__call__transformers.AlternatingCodebooksLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2327[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]

**Parameters:**

input_start_len (`int`) : The length of the initial input sequence.

semantic_vocab_size (`int`) : Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.

codebook_size (`int`) : Number of tokens associated to the codebook.

#### transformers.ClassifierFreeGuidanceLogitsProcessor[[transformers.ClassifierFreeGuidanceLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2234)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) for classifier free guidance (CFG). The scores are split over the batch dimension,
where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

See [the paper](https://huggingface.co/papers/2306.05284) for more information.

This logits processor is exclusively compatible with
[MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

Examples:

```python
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

__call__transformers.ClassifierFreeGuidanceLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2282[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

guidance_scale (float) : The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.EncoderNoRepeatNGramLogitsProcessor[[transformers.EncoderNoRepeatNGramLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1133)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that works similarly to [NoRepeatNGramLogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor), but applied exclusively to prevent
the repetition of n-grams present in the prompt.

It was designed to promote chattiness in a language model, by preventing the generation of n-grams present in
previous conversation rounds.

Examples:

```py
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

__call__transformers.EncoderNoRepeatNGramLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1182[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

encoder_ngram_size (`int`) : All ngrams of size `ngram_size` can only occur within the encoder input ids.

encoder_input_ids (`int`) : The encoder_input_ids that should not be repeated within the decoder ids.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.EncoderRepetitionPenaltyLogitsProcessor[[transformers.EncoderRepetitionPenaltyLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L409)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that works similarly to [RepetitionPenaltyLogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.RepetitionPenaltyLogitsProcessor), but with an *inverse* penalty
that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
selecting tokens that were present in the prompt.

It was designed to avoid hallucination in input-grounded tasks, like summarization. Although originally intended
for encoder-decoder models, it can also be used with decoder-only models like LLMs.

Examples:

```python
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

__call__transformers.EncoderRepetitionPenaltyLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L453[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

penalty (`float`) : The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0 and 1.0 penalizes prompt tokens.

encoder_input_ids (`torch.LongTensor`) : The encoder_input_ids that should be repeated within the decoder ids.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.EpsilonLogitsWarper[[transformers.EpsilonLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L854)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
Desmoothing](https://huggingface.co/papers/2210.15191) for more information.

Examples:
```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | 

>>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
>>> # Top P sampling, which restricts tokens based on their cumulative probability.
>>> # Pro tip: The paper recommends using `epsilon_cutoff` values between 3e-4 and 9e-4
>>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

__call__transformers.EpsilonLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L909[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

epsilon (`float`) : If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.

filter_value (`float`, *optional*, defaults to -inf) : All filtered values will be set to this float value.

min_tokens_to_keep (`int`, *optional*, defaults to 1) : Minimum number of tokens that cannot be filtered.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.EtaLogitsWarper[[transformers.EtaLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L923)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
min_tokens_to_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more information. Note: `do_sample`
must be set to `True` for this `LogitsProcessor` to work.

Examples:
```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | 

>>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
>>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
>>> # Pro tip: The paper recommends using `eta_cutoff` values between 3e-4 to 4e-3
>>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

__call__transformers.EtaLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L992[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

epsilon (`float`) : A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.

filter_value (`float`, *optional*, defaults to -inf) : All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This parameter is useful when logits need to be modified for very low probability tokens that should be excluded from generation entirely.

min_tokens_to_keep (`int`, *optional*, defaults to 1) : Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities. For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation, even if all tokens have probabilities below the cutoff `eta`.

device (`str`, *optional*, defaults to `"cpu"`) : The device to allocate the tensors.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.ExponentialDecayLengthPenalty[[transformers.ExponentialDecayLengthPenalty]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1795)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that exponentially increases the score of the `eos_token_id` after `start_index` has been
reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
predicted in a meaningful position.

Examples:

```python
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
which

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
published in 2010.
```

__call__transformers.ExponentialDecayLengthPenalty.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1881[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

exponential_decay_length_penalty (`tuple(int, float)`) : This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay

eos_token_id (`Union[int, list[int], torch.Tensor]`) : The id(s) of the *end-of-sequence* token.

input_ids_seq_length (`int`) : The length of the input sequence.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.ForcedBOSTokenLogitsProcessor[[transformers.ForcedBOSTokenLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1673)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that enforces the specified token as the first generated token. Used with encoder-decoder
models.

Examples:

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

>>> inputs = tokenizer("Translate from English to German: I love cats.", return_tensors="pt")

>>> # By default, it continues generating according to the model's logits
>>> outputs = model.generate(**inputs, max_new_tokens=10)
>>> print(tokenizer.batch_decode(outputs)[0])
 Ich liebe Kitty.

>>> # We can use `forced_bos_token_id` to force the start of generation with an encoder-decoder model
>>> # (including forcing it to end straight away with an EOS token)
>>> outputs = model.generate(**inputs, max_new_tokens=10, forced_bos_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(outputs)[0])

```

__call__transformers.ForcedBOSTokenLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1708[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

bos_token_id (`int`) : The id of the token to force as the first generated token.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.ForcedEOSTokenLogitsProcessor[[transformers.ForcedEOSTokenLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1718)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that enforces the specified token as the last generated token when `max_length` is reached.

Examples:

```python
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
A sequence: 1, 2, 3, 4, 5, 6, 7,
```

__call__transformers.ForcedEOSTokenLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1764[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

max_length (`int`) : The maximum length of the sequence to be generated.

eos_token_id (`Union[int, list[int], torch.Tensor]`) : The id(s) of the *end-of-sequence* token.

device (`str`, *optional*, defaults to `"cpu"`) : The device to allocate the tensors.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.InfNanRemoveLogitsProcessor[[transformers.InfNanRemoveLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1774)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
the logits processor should only be used if necessary since it can slow down the generation method.

This logits processor has no `generate` example, as there shouldn't be a correct combination of flags that warrants
its use.

__call__transformers.InfNanRemoveLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1783[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.LogitNormalization[[transformers.LogitNormalization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1896)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) for normalizing the scores using log-softmax. It's important to normalize
the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
the scores are normalized when comparing the hypotheses.

Examples:

```python
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

__call__transformers.LogitNormalization.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1927[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.LogitsProcessor[[transformers.LogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L50)

Abstract base class for all logit processors that can be applied during generation.

__call__transformers.LogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L53[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.LogitsProcessorList[[transformers.LogitsProcessorList]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L60)

This class can be used to create a list of [LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) to subsequently process a `scores` input tensor.
This class inherits from list and adds a specific *__call__* method to apply each [LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) to the
inputs.

__call__transformers.LogitsProcessorList.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L67[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
  beam search or log softmax for each vocabulary token when using beam search
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional kwargs that are specific to a logits processor.0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

kwargs (`dict[str, Any]`, *optional*) : Additional kwargs that are specific to a logits processor.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.MinLengthLogitsProcessor[[transformers.MinLengthLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L98)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
like most LLMs, the length includes the prompt.

Examples:

```python
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

__call__transformers.MinLengthLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L149[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

min_length (`int`) : The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.

eos_token_id (`Union[int, list[int], torch.Tensor]`) : The id(s) of the *end-of-sequence* token.

device (`str`, *optional*, defaults to `"cpu"`) : The device to allocate the tensors.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.MinNewTokensLengthLogitsProcessor[[transformers.MinNewTokensLengthLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L159)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
Contrarily to [MinLengthLogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.MinLengthLogitsProcessor), this processor ignores the prompt.

Examples:

```python
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

__call__transformers.MinNewTokensLengthLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L219[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

prompt_length_to_skip (`int`) : The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the input length.

min_new_tokens (`int`) : The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.

eos_token_id (`Union[int, list[int], torch.Tensor]`) : The id(s) of the *end-of-sequence* token.

device (`str`, *optional*, defaults to `"cpu"`) : The device to allocate the tensors.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.MinPLogitsWarper[[transformers.MinPLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L690)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that performs min-p, i.e. keeps all tokens that are above a minimum probability, scaled by the
probability of the most likely token. As a result, the filter becomes more aggressive in the presence of
high-probability tokens, which is a sign of a confident output that we shouldn't deviate from.

Often used together with [TemperatureLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TemperatureLogitsWarper). Used as an alternative to [TopPLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TopPLogitsWarper) and
[TopKLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TopKLogitsWarper).

Created by @menhguin and @kalomaze (github handles). Code adapted from [this external PR](https://github.com/oobabooga/text-generation-webui/pull/4449/files)

Examples:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | 

>>> # With `min_p` sampling, the output gets restricted to high-probability tokens.
>>> # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
>>> outputs = model.generate(**inputs, do_sample=True, min_p=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

__call__transformers.MinPLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L747[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]

**Parameters:**

min_p (`float`) : Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in the 0.99-0.8 range (use the opposite of normal `top_p` values).

filter_value (`float`, *optional*, defaults to -inf) : All filtered values will be set to this float value.

min_tokens_to_keep (`int`, *optional*, defaults to 1) : Minimum number of tokens that cannot be filtered.

#### transformers.NoBadWordsLogitsProcessor[[transformers.NoBadWordsLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1385)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that enforces that specified sequences will never be selected.

In order to get the token ids of the words that should not appear in the generated text, make sure to set
`add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
[here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

Examples:

```python
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

__call__transformers.NoBadWordsLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1277[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

bad_words_ids (`list[list[int]]`) : List of list of token ids that are not allowed to be generated.

eos_token_id (`Union[int, list[int], torch.Tensor]`, *optional*) : The id(s) of the *end-of-sequence* token.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.NoRepeatNGramLogitsProcessor[[transformers.NoRepeatNGramLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1074)

N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
avoiding repetitions of word sequences provides a more diverse output. This [LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) enforces no
repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
from consideration when further processing the scores. Note that, for decoder-only models like most LLMs, the
prompt is also considered to obtain the n-grams.
[Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
might lead to undesirable outcomes where the city's name appears only once in the entire text.
[Reference](https://huggingface.co/blog/how-to-generate)

Examples:

```py
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

__call__transformers.NoRepeatNGramLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1121[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

ngram_size (`int`) : All ngrams of size `ngram_size` can only occur once.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.PrefixConstrainedLogitsProcessor[[transformers.PrefixConstrainedLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1474)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that enforces constrained generation and is useful for prefix-conditioned constrained
generation. See [Autoregressive Entity Retrieval](https://huggingface.co/papers/2010.00904) for more information.

Examples:

```py
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

__call__transformers.PrefixConstrainedLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1525[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`) : This function constraints the beam search to allowed tokens only at each step. This function takes 2 arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID `batch_id`.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.RepetitionPenaltyLogitsProcessor[[transformers.RepetitionPenaltyLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L297)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that prevents the repetition of previous tokens through a penalty. This penalty is applied at
most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt
by default.

In the original [paper](https://huggingface.co/papers/1909.05858), the authors suggest the use of a penalty of around
1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

Examples:

```py
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

__call__transformers.RepetitionPenaltyLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L365[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

penalty (`float`) : The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated tokens. Between 0.0 and 1.0 rewards previously generated tokens.

prompt_ignore_length (`int`, *optional*) : The original input ids sequence length, which if provided, will not be used in the penalty calculation.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.SequenceBiasLogitsProcessor[[transformers.SequenceBiasLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1202)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that applies an additive bias on sequences. The bias is applied to the last token of a sequence
when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
one token, consider using beam methods (to gracefully work around partially completed sequences that have a
negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

At a token-level, biasing a word is different from biasing a word with a space before it. If you want to bias
"foo" mid-sentence, you'll likely want to add a prefix space and bias " foo" instead. Check the tokenizer section
of our NLP course to find out why: https://huggingface.co/learn/nlp-course/chapter2/4?fw=pt

Examples:

```python
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

__call__transformers.SequenceBiasLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1277[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

sequence_bias (`list[list[Union[list[int], float]]]`) : List of lists that maps a sequence of tokens to its bias term (e.g. `[[[10, 45], -2.0], [[64], -7.5]]`). Positive biases increase the odds of the sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be completed (in the token selection step after this processor is applied).

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.SuppressTokensAtBeginLogitsProcessor[[transformers.SuppressTokensAtBeginLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1933)

[SuppressTokensAtBeginLogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.SuppressTokensAtBeginLogitsProcessor) suppresses a list of tokens as soon as the `generate` function starts
generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
not generated at the beginning. Originally created for
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

Examples:

```python
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

__call__transformers.SuppressTokensAtBeginLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1975[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.SuppressTokensLogitsProcessor[[transformers.SuppressTokensLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1986)

This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so
that they are not generated. Originally created for
[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

Examples:

```python
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

__call__transformers.SuppressTokensLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2018[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.SynthIDTextWatermarkLogitsProcessor[[transformers.SynthIDTextWatermarkLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2681)

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

Refer to paper url: https://www.nature.com/articles/s41586-024-08025-4 for more details around this.

Examples:
```python
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

__call__transformers.SynthIDTextWatermarkLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2819[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

ngram_len (`int`) : Ngram length.

keys (`list[int]`) : A sequence of watermarking keys, one for each depth.

sampling_table_size (`int`) : Size of the sampling table.

sampling_table_seed (`int`) : Random seed to generate the sampling table.

context_history_size (`int`) : Size of the tensor to keep track of seen contexts.

device (`torch.device`) : Device to use.

skip_first_ngram_calls (`bool`, *optional*, defaults to `False`) : Whether to skip first ngram calls.

debug_mode (`bool`, optional, *optional*, defaults to `False`) : Logits are modified to uniform one got before watermarking modification is applied. This is to test the implementation.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.TemperatureLogitsWarper[[transformers.TemperatureLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L231)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) for temperature (exponential scaling output probability distribution), which effectively means
that it can control the randomness of the predicted tokens. Often used together with [TopPLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TopPLogitsWarper) and
[TopKLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TopKLogitsWarper).

Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
any effect.

Examples:

```python
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

__call__transformers.TemperatureLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L291[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

temperature (`float`) : Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely token.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.TopHLogitsWarper[[transformers.TopHLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L584)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that implements Top-H sampling, a decoding method which adaptively selects a subset of
high-probability tokens based on entropy and cumulative probability constraints.

This method dynamically determines how many tokens to keep by analyzing the entropy difference of the selected
distribution, thereby balancing exploration and exploitation. It ensures that generated text maintains both
diversity and coherence.

Reference:
For details, see *Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation*
(NeurIPS 2025): https://arxiv.org/abs/2509.02510

Example:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> outputs = model.generate(**inputs, do_sample=True, top_h=0.4)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

__call__transformers.TopHLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L635[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Input token IDs.
- **scores** (`torch.FloatTensor` of shape `(batch_size, vocab_size)`) --
  Raw logits from the model.0`torch.FloatTensor` of shape `(batch_size, vocab_size)`Processed logits where invalid tokens are masked with `-inf`.

Filters logits using Top-H sampling.

**Parameters:**

top_h (`float`) : Scaling coefficient for the entropy-based threshold (`tau`). Must be in the range `(0, 1]`. 

filter_value (`float`, *optional*, defaults to -inf) : All filtered values will be set to this float value.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, vocab_size)``

Processed logits where invalid tokens are masked with `-inf`.

#### transformers.TopKLogitsWarper[[transformers.TopKLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L531)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that performs top-k, i.e. restricting to the k highest probability elements. Often used
together with [TemperatureLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TemperatureLogitsWarper) and [TopPLogitsWarper](/docs/transformers/main/en/internal/generation_utils#transformers.TopPLogitsWarper).

Examples:

```python
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

__call__transformers.TopKLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L575[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

top_k (`int`) : The number of highest probability vocabulary tokens to keep for top-k-filtering.

filter_value (`float`, *optional*, defaults to -inf) : All filtered values will be set to this float value.

min_tokens_to_keep (`int`, *optional*, defaults to 1) : Minimum number of tokens that cannot be filtered.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.TopPLogitsWarper[[transformers.TopPLogitsWarper]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L464)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that performs top-p, i.e. restricting to top tokens summing to prob_cut_off 

Examples:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3 | 

>>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
>>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
>>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

__call__transformers.TopPLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L515[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

top_p (`float`) : If set to >> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

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

__call__transformers.TypicalLogitsWarper.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L830[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

mass (`float`, *optional*, defaults to 0.9) : Value of typical_p between 0 and 1 inclusive, defaults to 0.9.

filter_value (`float`, *optional*, defaults to -inf) : All filtered values will be set to this float value.

min_tokens_to_keep (`int`, *optional*, defaults to 1) : Minimum number of tokens that cannot be filtered.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor[[transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2343)

Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

See [the paper](https://huggingface.co/papers/2306.17806) for more information.

Examples:

```python
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

__call__transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2449[{"name": "input_ids", "val": ""}, {"name": "scores", "val": ""}]

**Parameters:**

guidance_scale (`float`) : The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.

model (`PreTrainedModel`) : The model computing the unconditional scores. Supposedly the same as the one computing the conditional scores. Both models must use the same tokenizer.

unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to the last token of the prompt.

unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Attention mask for unconditional_ids.

use_cache (`bool`, *optional*, defaults to `True`) : Whether to cache key/values during the negative prompt forward pass.

#### transformers.WhisperTimeStampLogitsProcessor[[transformers.WhisperTimeStampLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2026)

[LogitsProcessor](/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor) that modifies the logits for the generation of timestamps in the transcription. When the input
tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
potential tokens.

See [the paper](https://huggingface.co/papers/2212.04356) for more information.

Examples:
``` python
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
Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca.

>>> #No timestamps & change EOS:
>>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
>>> model.generation_config.eos_token_id = 460
>>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print("Transcription:", transcription)
Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can
```

__call__transformers.WhisperTimeStampLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2115[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

generate_config (`GenerateConfig`) : The generate config used to generate the output. The following parameters are required: eos_token_id (`int`, *optional*, defaults to 50257): The id of the *end-of-sequence* token. no_timestamps_token_id (`int`, *optional*, defaults to 50363): The id of the `""` token. max_initial_timestamp_index (`int`, *optional*, defaults to 1): Used to set the maximum value of the initial timestamp. This is used to prevent the model from predicting timestamps that are too far in the future.

begin_index (`int`) : Token index of the first token that is generated by the model.

_detect_timestamp_from_logprob (`bool`, *optional*) : Whether timestamps can be predicted from logprobs over all timestamps.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

#### transformers.WatermarkLogitsProcessor[[transformers.WatermarkLogitsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2508)

Logits processor for watermarking generated text. The processor modifies model output scores by adding a small bias to
randomized set of "green" tokens before generating the next token. "Green" tokens selection process depends on the
`seeding_scheme` used. The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

The text generated by this `LogitsProcessor` can be detected using `WatermarkDetector`. See [__call__()](/docs/transformers/main/en/internal/generation_utils#transformers.WatermarkDetector.__call__) for details,

See [the paper](https://huggingface.co/papers/2306.04634) for more information.

Examples:

```python
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

__call__transformers.WatermarkLogitsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L2630[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
  search or log softmax for each vocabulary token when using beam search0`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`The processed prediction scores.

**Parameters:**

vocab_size (`int`) : The model tokenizer's vocab_size. Used to calculate "green" tokens ratio.

device (`str`) : The device where model is allocated.

greenlist_ratio (`float`, optional, *optional*, defaults to 0.25) : The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.

bias (`float`, optional, *optional*, defaults to 2.0) : The bias added to the selected "green" tokens' logits. Consider lowering the `bias` if the text generation quality degrades. Recommended values are in the range of [0.5, 2.0]. Defaults to 2.0.

hashing_key (`int`, optional, *optional*, defaults to 15485863) : Key used for hashing. If you deploy this watermark, we advise using another private key. Defaults to 15485863 (the millionth prime).

seeding_scheme (`str`, optional, *optional*, defaults to `"lefthash"`) : The seeding scheme used for selecting "green" tokens. Accepts values: - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from paper) - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from paper) The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash". The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.

context_width (`int`, *optional*, defaults to 1) : The number of previous tokens to use when setting the seed.

**Returns:**

``torch.FloatTensor` of shape `(batch_size, config.vocab_size)``

The processed prediction scores.

## StoppingCriteria[[transformers.StoppingCriteria]]

A [StoppingCriteria](/docs/transformers/main/en/internal/generation_utils#transformers.StoppingCriteria) can be used to change when to stop generation (other than EOS token). Please note that this is exclusively available to our PyTorch implementations.

#### transformers.StoppingCriteria[[transformers.StoppingCriteria]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L46)

Abstract base class for all stopping criteria that can be applied during generation.

If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
output_scores=True` to `generate`.

__call__transformers.StoppingCriteria.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L53[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional stopping criteria specific kwargs.0`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.

kwargs (`dict[str, Any]`, *optional*) : Additional stopping criteria specific kwargs.

**Returns:**

``torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

#### transformers.StoppingCriteriaList[[transformers.StoppingCriteriaList]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L496)

__call__transformers.StoppingCriteriaList.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L497[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional stopping criteria specific kwargs.0`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) : Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.

kwargs (`dict[str, Any]`, *optional*) : Additional stopping criteria specific kwargs.

**Returns:**

``torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

#### transformers.MaxLengthCriteria[[transformers.MaxLengthCriteria]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L58)

This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
in mind for decoder-only type of transformers, this will include the initial prompted tokens.

__call__transformers.MaxLengthCriteria.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L74[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional stopping criteria specific kwargs.0`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

**Parameters:**

max_length (`int`) : The maximum length that the output sequence can have in number of tokens.

max_position_embeddings (`int`, *optional*) : The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.

**Returns:**

``torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

#### transformers.MaxTimeCriteria[[transformers.MaxTimeCriteria]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L87)

This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
time will start being counted when you initialize this function. You can override this by passing an
`initial_time`.

__call__transformers.MaxTimeCriteria.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L104[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional stopping criteria specific kwargs.0`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

**Parameters:**

max_time (`float`) : The maximum allowed time in seconds for the generation.

initial_time (`float`, *optional*, defaults to `time.time()`) : The start of the generation allowed time.

**Returns:**

``torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

#### transformers.StopStringCriteria[[transformers.StopStringCriteria]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L110)

This class can be used to stop generation whenever specific string sequences are generated. It preprocesses
the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

Generation is stopped as soon as a token is generated that completes any of the stop strings.
We want to catch any instance in which the stop string would be present in the decoded output, which means
we must also catch cases with "overhangs" off one or both ends. To make this more concrete, for the stop string
"stop", any of the following token sequences would trigger the match:

- ["st", "op"]
- ["stop"]
- ["st", "opera"]
- ["sto", "pper"]
- ["las", "topper"]
- ["s", "to", "pped"]

Note that a match will only be triggered if the stop string is at the end of the generated sequence. In other
words, these sequences will not trigger a match:

- ["stop", "at"]
- ["st", "op", "at"]
- ["st", "opera", "tion"]

The reason these are not a match is that the stop string does not overlap with the final token. If you can remove
one or more tokens from the end of the sequence without destroying the stop string, then this criterion will not
match that stop string. This is by design; because this check is run after each token is generated, we can't miss a
valid stop string if one is generated, but we don't want to halt generation just because the stop string exists
somewhere in the past input_ids.

How is the match actually performed, though? We do it in quite a confusing way, because we want the entire match
process to be compilable with Torch or XLA, which means we cannot use standard string methods. However, it is possible,
with some work, to do string matching with pure tensor operations. We'll begin by describing the algorithm we use
with standard string operations, and then at the end we'll explain how this is converted to pure tensor operations.

The key to the algorithm is an observation: Because the stop string must overlap with the end of the token sequence, we can start at
the end of the sequence and work backwards. Specifically, we check that there is an overlap between the start of
the final token and the end of the stop_string, or to put it another way, stop_string[-i:] == token[:i] for
some i > 0. If you look at the positive examples above, you'll see the last token in all of them fulfills this
property:

- ["st", "op"] (overlap is "op", overlap length == 2)
- ["stop"]  (overlap is "stop", overlap length == 4)
- ["st", "opera"]  (overlap is "op", overlap length == 2)
- ["sto", "pper"]  (overlap is "p", overlap length == 1)
- ["las", "topper"]  (overlap is "top", overlap length == 3)
- ["s", "to", "pped"]  (overlap is "p", overlap length == 1)

It's impossible to construct a matching sequence that does not have this property (feel free to verify this
yourself). However, although this overlap between the start of the final token and the end of the stop string is
necessary for a match, it is not sufficient. We also need to check that the rest of the token sequence is
consistent with the stop string.

How do we do that? Let's use ["s", "to", "pped"] as an example. We know that the final token, "pped", has an
overlap of 1 with the stop string, "stop". We then go back to the previous token, "to". Since we have already
matched 1 character from the stop string, the remainder to check is "sto". We check that the next token "to"
matches the end of the remainder, which it does. We have now matched 3 characters from the stop string, and the
remainder to match is "s". We go back to the previous token again, which is also "s". This is a match, and so
we have matched the entire stop string.

How does it work when the tokens run off the start of the stop string, though? Let's consider the example of
["las", "topper"]. The final token, "topper", has an overlap of 3 with the stop string, "stop". Therefore,
the remaining stop string to match is "s". We go back to the previous token, "las". Because the remainder to
match is just "s", with length 1, we consider only the final 1 character from the token, which is "s". This
matches the stop string, and so the entire string is matched.

How do we compute these matches with tensor operations, though? Simply: we efficiently precompute the necessary
information for all tokens! For every token, we compute:
- Its overlap with the end of the stop string, if any
- The positions inside the stop string where the token matches, including matches that run off the start.
- The total length of the token

For example, for the token "pped", we would compute an end overlap of 1, no internal matching positions,
and a length of 4. For the token "to", we would compute no end overlap, a single internal matching position
of 1 (counting from the end), and a length of 2. For the token "s", we would compute no end overlap,
a single internal matching position of 3 (again counting from the end) and a length of 1.

As long as we have this information, we can execute the algorithm above without any string comparison
operations. We simply perform the following steps:
- Check if the final token has an end-overlap with the start string
- Continue backwards, keeping track of how much of the stop string we've matched so far
- At each point, check if the next token has the current position as one of its valid positions
- Continue until either a match fails, or we completely match the whole stop string

Again, consider ["s", "to", "pped"] as an example. "pped" has an end overlap of 1, so we can begin a match.
We have matched 1 character so far, so we check that the next token "to", has 1 as a valid position (again,
counting from the end). It does, so we add the length of "to" to our position tracker. We have now matched
3 characters, so we check that the next token "s" has 3 as a valid position. It does, so we add its length
to the position tracker. The position tracker is now 4, which is the length of the stop string. We have matched the
entire stop string.

In the second case, ["las", "topper"], "topper" has an end overlap of 3, so we can begin a match. We have
matched 3 characters so far, so we check that the next token "las" has 3 as a valid position. It does, because we
allow tokens to match positions that run off the start of the stop string. We add its length to the position
tracker. The position tracker is now 6, which is greater than the length of the stop string! Don't panic, though -
this also counts as a match of the stop string. We have matched the entire stop string.

Examples:

```python
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

__call__transformers.StopStringCriteria.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L389[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional stopping criteria specific kwargs.0`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

**Parameters:**

tokenizer (`PreTrainedTokenizer`) : The model's associated tokenizer (necessary to extract vocab and tokenize the termination sequences)

stop_strings (`Union[str, list[str]]`) : A list of strings that should end generation. If a string is passed, it will be treated like a list with a single element.

**Returns:**

``torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

#### transformers.EosTokenCriteria[[transformers.EosTokenCriteria]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L451)

This class can be used to stop generation whenever the "end-of-sequence" token is generated.
By default, it uses the `model.generation_config.eos_token_id`.

__call__transformers.EosTokenCriteria.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L468[{"name": "input_ids", "val": ": LongTensor"}, {"name": "scores", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) --
  Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
  make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional stopping criteria specific kwargs.0`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

**Parameters:**

eos_token_id (`Union[int, list[int], torch.Tensor]`) : The id(s) of the *end-of-sequence* token.

**Returns:**

``torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`)`

`True` indicates we stop generation for a particular row.
`False` indicates we should continue.

## Streamers[[transformers.TextStreamer]]

#### transformers.TextStreamer[[transformers.TextStreamer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L41)

Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

The API for the streamer classes is still under development and may change in the future.

Examples:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

endtransformers.TextStreamer.endhttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L119[]
Flushes any remaining cache and prints a newline to stdout.

**Parameters:**

tokenizer (`AutoTokenizer`) : The tokenized used to decode the tokens.

skip_prompt (`bool`, *optional*, defaults to `False`) : Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.

decode_kwargs (`dict`, *optional*) : Additional keyword arguments to pass to the tokenizer's `decode` method.
#### on_finalized_text[[transformers.TextStreamer.on_finalized_text]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L133)

Prints the new text to stdout. If the stream is ending, also prints a newline.
#### put[[transformers.TextStreamer.put]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L85)

Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.

#### transformers.TextIteratorStreamer[[transformers.TextIteratorStreamer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L162)

Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
useful for applications that benefit from accessing the generated text in a non-blocking way (e.g. in an interactive
Gradio demo).

The API for the streamer classes is still under development and may change in the future.

Examples:

```python
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

on_finalized_texttransformers.TextIteratorStreamer.on_finalized_texthttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L216[{"name": "text", "val": ": str"}, {"name": "stream_end", "val": ": bool = False"}]
Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.

**Parameters:**

tokenizer (`AutoTokenizer`) : The tokenized used to decode the tokens.

skip_prompt (`bool`, *optional*, defaults to `False`) : Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.

timeout (`float`, *optional*) : The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions in `.generate()`, when it is called in a separate thread.

decode_kwargs (`dict`, *optional*) : Additional keyword arguments to pass to the tokenizer's `decode` method.

#### transformers.AsyncTextIteratorStreamer[[transformers.AsyncTextIteratorStreamer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L233)

Streamer that stores print-ready text in a queue, to be used by a downstream application as an async iterator.
This is useful for applications that benefit from accessing the generated text asynchronously (e.g. in an
interactive Gradio demo).

The API for the streamer classes is still under development and may change in the future.

Examples:

```python
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

on_finalized_texttransformers.AsyncTextIteratorStreamer.on_finalized_texthttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L296[{"name": "text", "val": ": str"}, {"name": "stream_end", "val": ": bool = False"}]
Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.

**Parameters:**

tokenizer (`AutoTokenizer`) : The tokenized used to decode the tokens.

skip_prompt (`bool`, *optional*, defaults to `False`) : Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.

timeout (`float`, *optional*) : The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions in `.generate()`, when it is called in a separate thread.

decode_kwargs (`dict`, *optional*) : Additional keyword arguments to pass to the tokenizer's `decode` method.

## Caches[[transformers.CacheLayerMixin]]

#### transformers.CacheLayerMixin[[transformers.CacheLayerMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L26)

Base, abstract class for a single layer's cache.

updatetransformers.CacheLayerMixin.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L42[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]
#### get_seq_length[[transformers.CacheLayerMixin.get_seq_length]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L50)
#### get_mask_sizes[[transformers.CacheLayerMixin.get_mask_sizes]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L47)
#### get_max_cache_shape[[transformers.CacheLayerMixin.get_max_cache_shape]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L53)
#### reset[[transformers.CacheLayerMixin.reset]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L68)

Resets the cache values while preserving the objects
#### reorder_cache[[transformers.CacheLayerMixin.reorder_cache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L77)

Reorders this layer's cache for beam search.
#### lazy_initialization[[transformers.CacheLayerMixin.lazy_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L39)

#### transformers.DynamicLayer[[transformers.DynamicLayer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L84)

A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.

updatetransformers.DynamicLayer.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L98[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]- **key_states** (`torch.Tensor`) -- The new key states to cache.
- **value_states** (`torch.Tensor`) -- The new value states to cache.
- **cache_kwargs** (`dict[str, Any]`, *optional*) -- Additional arguments for the cache.0tuple[`torch.Tensor`, `torch.Tensor`]The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

**Parameters:**

key_states (`torch.Tensor`) : The new key states to cache.

value_states (`torch.Tensor`) : The new value states to cache.

cache_kwargs (`dict[str, Any]`, *optional*) : Additional arguments for the cache.

**Returns:**

`tuple[`torch.Tensor`, `torch.Tensor`]`

The key and value states.
#### lazy_initialization[[transformers.DynamicLayer.lazy_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L92)
#### crop[[transformers.DynamicLayer.crop]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L140)

Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
to remove `max_length` tokens.
#### batch_repeat_interleave[[transformers.DynamicLayer.batch_repeat_interleave]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L154)

Repeat the cache `repeats` times in the batch dimension.
#### batch_select_indices[[transformers.DynamicLayer.batch_select_indices]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L160)

Only keep the `indices` in the batch dimension of the cache.

#### transformers.StaticLayer[[transformers.StaticLayer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L253)

A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

updatetransformers.StaticLayer.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L307[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]- **key_states** (`torch.Tensor`) -- The new key states to cache.
- **value_states** (`torch.Tensor`) -- The new value states to cache.
- **cache_kwargs** (`dict[str, Any]`, *optional*) -- Additional arguments for the cache.0tuple[`torch.Tensor`, `torch.Tensor`]The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

**Parameters:**

max_cache_len (`int`) : Maximum number of tokens that can be stored, used for tensor preallocation.

**Returns:**

`tuple[`torch.Tensor`, `torch.Tensor`]`

The key and value states.
#### lazy_initialization[[transformers.StaticLayer.lazy_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L270)

Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
internally don't compile the prefill, this is guaranteed to have been called already when compiling.
If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
not be compiled anyway for performances!

#### transformers.StaticSlidingWindowLayer[[transformers.StaticSlidingWindowLayer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L362)

A static cache layer that stores the key and value states as static tensors of shape
`[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing
tensors, and then mutates them in-place. Built for `torch.compile` support.

updatetransformers.StaticSlidingWindowLayer.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L382[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]- **key_states** (`torch.Tensor`) -- The new key states to cache.
- **value_states** (`torch.Tensor`) -- The new value states to cache.
- **cache_kwargs** (`dict[str, Any]`, *optional*) -- Additional arguments for the cache.0tuple[`torch.Tensor`, `torch.Tensor`]The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

**Parameters:**

max_cache_len (`int`) : Maximum number of tokens that can be stored, used for tensor preallocation.

sliding_window (`int`) : The size of the sliding window.

**Returns:**

`tuple[`torch.Tensor`, `torch.Tensor`]`

The key and value states.
#### lazy_initialization[[transformers.StaticSlidingWindowLayer.lazy_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L270)

Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
internally don't compile the prefill, this is guaranteed to have been called already when compiling.
If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
not be compiled anyway for performances!

#### transformers.QuantoQuantizedLayer[[transformers.QuantoQuantizedLayer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L567)

updatetransformers.QuantoQuantizedLayer.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L515[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]- **key_states** (`torch.Tensor`) -- The new key states to cache.
- **value_states** (`torch.Tensor`) -- The new value states to cache.
- **cache_kwargs** (`dict[str, Any]`, *optional*) -- Additional arguments for the cache.0tuple[`torch.Tensor`, `torch.Tensor`]The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

**Parameters:**

key_states (`torch.Tensor`) : The new key states to cache.

value_states (`torch.Tensor`) : The new value states to cache.

cache_kwargs (`dict[str, Any]`, *optional*) : Additional arguments for the cache.

**Returns:**

`tuple[`torch.Tensor`, `torch.Tensor`]`

The key and value states.
#### lazy_initialization[[transformers.QuantoQuantizedLayer.lazy_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L92)

#### transformers.HQQQuantizedLayer[[transformers.HQQQuantizedLayer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L617)

updatetransformers.HQQQuantizedLayer.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L515[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]- **key_states** (`torch.Tensor`) -- The new key states to cache.
- **value_states** (`torch.Tensor`) -- The new value states to cache.
- **cache_kwargs** (`dict[str, Any]`, *optional*) -- Additional arguments for the cache.0tuple[`torch.Tensor`, `torch.Tensor`]The key and value states.

Update the key and value caches in-place, and return the necessary keys and value states.

**Parameters:**

key_states (`torch.Tensor`) : The new key states to cache.

value_states (`torch.Tensor`) : The new value states to cache.

cache_kwargs (`dict[str, Any]`, *optional*) : Additional arguments for the cache.

**Returns:**

`tuple[`torch.Tensor`, `torch.Tensor`]`

The key and value states.
#### lazy_initialization[[transformers.HQQQuantizedLayer.lazy_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L92)

#### transformers.Cache[[transformers.Cache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L671)

A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
the Cache of each layer.

updatetransformers.Cache.updatehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L747[{"name": "key_states", "val": ": Tensor"}, {"name": "value_states", "val": ": Tensor"}, {"name": "layer_idx", "val": ": int"}, {"name": "cache_kwargs", "val": ": typing.Optional[dict[str, typing.Any]] = None"}]- **key_states** (`torch.Tensor`) --
  The new key states to cache.
- **value_states** (`torch.Tensor`) --
  The new value states to cache.
- **layer_idx** (`int`) --
  The index of the layer to cache the states for.
- **cache_kwargs** (`dict[str, Any]`, *optional*) --
  Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
  cache to be created.0A tuple containing the updated key and value states.

Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

**Parameters:**

layers (`Optional`, *optional*) : A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will be used.

layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*) : Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer, and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current list of layers.

offloading (`bool`, *optional*, defaults to `False`) : Whether to perform offloading of the layers to `cpu`, to save GPU memory.

offload_only_non_sliding (`bool`, *optional*, defaults to `True`) : If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

**Returns:**

A tuple containing the updated key and value states.
#### early_initialization[[transformers.Cache.early_initialization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L788)

Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
This is useful for our `export` recipes, as `export` needs everything in advance.
#### get_seq_length[[transformers.Cache.get_seq_length]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L803)

Returns the sequence length of the cache for the given layer.
#### get_mask_sizes[[transformers.Cache.get_mask_sizes]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L809)

Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
the given layer at `layer_idx`.
The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
#### get_max_cache_shape[[transformers.Cache.get_max_cache_shape]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L821)

Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length.
#### reset[[transformers.Cache.reset]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L829)

Recursively reset all layers tensors
#### reorder_cache[[transformers.Cache.reorder_cache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L834)

Reorder the cache for beam search
#### crop[[transformers.Cache.crop]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L839)

Crop the cache to the given length
#### batch_repeat_interleave[[transformers.Cache.batch_repeat_interleave]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L844)

Repeat and interleave the cache
#### batch_select_indices[[transformers.Cache.batch_select_indices]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L849)

Select indices from the cache

#### transformers.DynamicCache[[transformers.DynamicCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L895)

A cache that grows dynamically as more tokens are generated. This is the default for generative models.
It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

See `Cache` for details on common methods that are implemented by all cache classes.

Example:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

>>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

>>> # Prepare a cache class and pass it to model's forward
>>> past_key_values = DynamicCache(config=model.config)
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values # access cache filled with key/values from generation
```

**Parameters:**

ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*) : It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]). Note: it needs to be the 1st arg as well to work correctly

config (`PreTrainedConfig`, *optional*) : The config of the model for which this Cache will be used. If passed, it will be used to check for sliding or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

offloading (`bool`, *optional*, defaults to `False`) : Whether to perform offloading of the layers to `cpu`, to save GPU memory.

offload_only_non_sliding (`bool`, *optional*, defaults to `False`) : If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

#### transformers.StaticCache[[transformers.StaticCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1004)

Static Cache class to be used with `torch.compile(model)` and `torch.export()`. It will check the `config`
for potential hybrid cache structure, and initialize each layer accordingly.

See `Cache` for details on common methods that are implemented by all cache classes.

Example:

```python
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

**Parameters:**

config (`PreTrainedConfig`) : The config of the model for which this Cache will be used. It will be used to check for sliding or hybrid layer structure, and initialize each layer accordingly.

max_cache_len (`int`) : The maximum number of tokens that this Cache should hold.

offloading (`bool`, *optional*, defaults to `False`) : Whether to perform offloading of the layers to `cpu`, to save GPU memory.

offload_only_non_sliding (`bool`, *optional*, defaults to `True`) : If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

#### transformers.QuantizedCache[[transformers.QuantizedCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1083)

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

**Parameters:**

backend (`str`) : The quantization backend to use. One of `("quanto", "hqq").

config (`PreTrainedConfig`) : The config of the model for which this Cache will be used.

nbits (`int`, *optional*, defaults to 4) : The number of bits for quantization.

axis_key (`int`, *optional*, defaults to 0) : The axis on which to quantize the keys.

axis_value (`int`, *optional*, defaults to 0) : The axis on which to quantize the values.

q_group_size (`int`, *optional*, defaults to 64) : Quantization is done per-channel according to a set `q_group_size` for both keys and values.

residual_length (`int`, *optional*, defaults to 128) : Maximum capacity for the original precision cache

#### transformers.EncoderDecoderCache[[transformers.EncoderDecoderCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1139)

Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
cross-attention caches.

See `Cache` for details on common methods that are implemented by all cache classes.

Example:

```python
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

batch_repeat_interleavetransformers.EncoderDecoderCache.batch_repeat_interleavehttps://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1268[{"name": "repeats", "val": ": int"}]
Repeat the cache `repeats` times in the batch dimension. Used in contrastive search (on the Hub).

**Parameters:**

caches (`Iterable`) : Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the second one for cross-attention. Can optionally also be an iterable of length 1, containing a `tuple[tuple[torch.Tensor]]` (usually used for compatibility with torch dp and ddp).
#### batch_select_indices[[transformers.EncoderDecoderCache.batch_select_indices]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1274)

Only keep the `indices` in the batch dimension of the cache. Used in contrastive search (on the Hub).
#### batch_split[[transformers.EncoderDecoderCache.batch_split]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1254)

Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
`_split_model_inputs()` in `generation.utils`
#### crop[[transformers.EncoderDecoderCache.crop]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1246)

Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search (on the Hub).
#### get_max_cache_shape[[transformers.EncoderDecoderCache.get_max_cache_shape]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1280)

Returns the maximum sequence length (i.e. max capacity) of the cache object
#### get_seq_length[[transformers.EncoderDecoderCache.get_seq_length]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1220)

Returns the sequence length of the cached states. A layer index can be optionally passed.
#### reorder_cache[[transformers.EncoderDecoderCache.reorder_cache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1230)

Reorders the cache for beam search, given the selected beam indices.

#### transformers.QuantoQuantizedCache[[transformers.QuantoQuantizedCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1371)

#### transformers.HQQQuantizedCache[[transformers.HQQQuantizedCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1388)

#### transformers.OffloadedCache[[transformers.OffloadedCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1317)

#### transformers.OffloadedStaticCache[[transformers.OffloadedStaticCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1326)

#### transformers.HybridCache[[transformers.HybridCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1344)

#### transformers.HybridChunkedCache[[transformers.HybridChunkedCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1353)

#### transformers.SlidingWindowCache[[transformers.SlidingWindowCache]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L1335)

## Watermark Utils[[transformers.WatermarkingConfig]]

#### transformers.WatermarkingConfig[[transformers.WatermarkingConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L1258)

Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
See [this paper](https://huggingface.co/papers/2306.04634) for more details on the arguments.

Accepts the following keys:
- greenlist_ratio (`float`):
  Used for watermarking. The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
- bias (`float`):
  Used with watermarking. The bias added to the selected "green" tokens' logits. Defaults to 2.0.
- hashing_key (`int`):
  Hashing key used for watermarking. Defaults to 15485863 (the millionth prime).
- seeding_scheme (`str`):
  Algorithm to use for watermarking. Accepts values:
  - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from the paper)
  - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from the paper)
    The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
- context_width(`int`):
  The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.

__call__transformers.WatermarkingConfig.__call__[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]
Call self as a function.

#### transformers.WatermarkDetector[[transformers.WatermarkDetector]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L69)

Detector for detection of watermark generated text. The detector needs to be given the exact same settings that were
given during text generation to replicate the watermark greenlist generation and so detect the watermark. This includes
the correct device that was used during text generation, the correct watermarking arguments and the correct tokenizer vocab size.
The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

See [the paper](https://huggingface.co/papers/2306.04634) for more information.

Examples:

```python
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

__call__transformers.WatermarkDetector.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L189[{"name": "input_ids", "val": ": LongTensor"}, {"name": "z_threshold", "val": ": float = 3.0"}, {"name": "return_dict", "val": ": bool = False"}]- **input_ids** (`torch.LongTensor`) --
  The watermark generated text. It is advised to remove the prompt, which can affect the detection.
- **z_threshold** (`Dict`, *optional*, defaults to `3.0`) --
  Changing this threshold will change the sensitivity of the detector. Higher z threshold gives less
  sensitivity and vice versa for lower z threshold.
- **return_dict** (`bool`,  *optional*, defaults to `False`) --
  Whether to return `~generation.WatermarkDetectorOutput` or not. If not it will return boolean predictions,0`WatermarkDetectorOutput` or `np.ndarray`A `WatermarkDetectorOutput`
if `return_dict=True` otherwise a `np.ndarray`.

ma

**Parameters:**

model_config (`PreTrainedConfig`) : The model config that will be used to get model specific arguments used when generating.

device (`str`) : The device which was used during watermarked text generation.

watermarking_config (Union[`WatermarkingConfig`, `Dict`]) : The exact same watermarking config and arguments used when generating text.

ignore_repeated_ngrams (`bool`, *optional*, defaults to `False`) : Whether to count every unique ngram only once or not.

max_cache_size (`int`, *optional*, defaults to 128) : The max size to be used for LRU caching of seeding/sampling algorithms called for every token.

**Returns:**

``WatermarkDetectorOutput` or `np.ndarray``

A `WatermarkDetectorOutput`
if `return_dict=True` otherwise a `np.ndarray`.

#### transformers.BayesianDetectorConfig[[transformers.BayesianDetectorConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L241)

This is the configuration class to store the configuration of a [BayesianDetectorModel](/docs/transformers/main/en/internal/generation_utils#transformers.BayesianDetectorModel). It is used to
instantiate a Bayesian Detector model according to the specified arguments.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

watermarking_depth (`int`, *optional*) : The number of tournament layers.

base_rate (`float1`, *optional*, defaults to 0.5) : Prior probability P(w) that a text is watermarked.

#### transformers.BayesianDetectorModel[[transformers.BayesianDetectorModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L348)

Bayesian classifier for watermark detection.

This detector uses Bayes' rule to compute a watermarking score, which is the sigmoid of the log of ratio of the
posterior probabilities P(watermarked|g_values) and P(unwatermarked|g_values). Please see the section on
BayesianScore in the paper for further details.
Paper URL: https://www.nature.com/articles/s41586-024-08025-4

Note that this detector only works with non-distortionary Tournament-based watermarking using the Bernoulli(0.5)
g-value distribution.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.BayesianDetectorModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L435[{"name": "g_values", "val": ": Tensor"}, {"name": "mask", "val": ": Tensor"}, {"name": "labels", "val": ": torch.Tensor | None = None"}, {"name": "loss_batch_weight", "val": " = 1"}, {"name": "return_dict", "val": " = False"}]- **g_values** (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth, ...)`) --
  g-values (with values 0 or 1)
- **mask** --
  A binary array shape [batch_size, seq_len] indicating which g-values should be used. g-values with mask
  value 0 are discarded.0p(watermarked | g_values), of shape [batch_size].

Computes the watermarked posterior P(watermarked|g_values).

**Parameters:**

config ([BayesianDetectorConfig](/docs/transformers/main/en/internal/generation_utils#transformers.BayesianDetectorConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

p(watermarked | g_values), of shape [batch_size].

#### transformers.SynthIDTextWatermarkingConfig[[transformers.SynthIDTextWatermarkingConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L1336)

Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
See [this paper](https://www.nature.com/articles/s41586-024-08025-4) for more details on the arguments.

Examples:
```python
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

**Parameters:**

ngram_len (`int`) : Ngram length.

keys (`list[int]`) : A sequence of watermarking keys, one for each depth.

context_history_size (`int`, *optional*, defaults to 1024) : Size of the tensor to keep track of seen contexts.

sampling_table_seed (`int`, *optional*, defaults to 0) : Random seed to generate the sampling table.

sampling_table_size (`int`, *optional*, defaults to 65536) : Size of the sampling table.

skip_first_ngram_calls (`bool`, *optional*, defaults to `False`) : Whether to skip first ngram calls.

debug_mode (`bool`, optional, *optional*, defaults to `False`) : Logits are modified to uniform one got before watermarking modification is applied. This is to test the implementation.

#### transformers.SynthIDTextWatermarkDetector[[transformers.SynthIDTextWatermarkDetector]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L479)

SynthID text watermark detector class.

This class has to be initialized with the trained bayesian detector module check script
in examples/synthid_text/detector_training.py for example in training/saving/loading this
detector module. The folder also showcases example use case of this detector.

Examples:
```python
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

__call__transformers.SynthIDTextWatermarkDetector.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/generation/watermarking.py#L526[{"name": "tokenized_outputs", "val": ": Tensor"}]

**Parameters:**

detector_module ([BayesianDetectorModel](/docs/transformers/main/en/internal/generation_utils#transformers.BayesianDetectorModel)) : Bayesian detector module object initialized with parameters. Check https://github.com/huggingface/transformers-research-projects/tree/main/synthid_text for usage.

logits_processor (`SynthIDTextWatermarkLogitsProcessor`) : The logits processor used for watermarking.

tokenizer (`Any`) : The tokenizer used for the model.

## Compile Utils[[transformers.CompileConfig]]

#### transformers.CompileConfig[[transformers.CompileConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L1426)

Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details on the arguments.

Examples:
```python
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

__call__transformers.CompileConfig.__call__[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]
Call self as a function.

**Parameters:**

fullgraph (`bool`, *optional*, defaults to `False`) : If False (default), attempts to discover compileable regions that will be optimized. If True, then require that the entire function be capturable into a single graph. If this is not possible (that is, if there are graph breaks), then an error will be raised.

dynamic (`bool` or `None`, *optional*) : Whether to try to use dynamic shape graphs.

backend (`str` or `Callable`, *optional*, defaults to `"inductor"`) : Backend to be used.

mode (`str`, *optional*, defaults to `"reduce-overhead"`) : Controls balance between performance and overhead.

options (`dict`, *optional*) : A dictionary of options to pass to the backend.
