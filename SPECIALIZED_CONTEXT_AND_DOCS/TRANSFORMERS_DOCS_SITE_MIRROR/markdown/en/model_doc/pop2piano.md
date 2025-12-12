# Pop2Piano

## Overview

The Pop2Piano model was proposed in [Pop2Piano : Pop Audio-based Piano Cover Generation](https://huggingface.co/papers/2211.00895) by Jongho Choi and Kyogu Lee.

Piano covers of pop music are widely enjoyed, but generating them from music is not a trivial task. It requires great
expertise with playing piano as well as knowing different characteristics and melodies of a song. With Pop2Piano you
can directly generate a cover from a song's audio waveform. It is the first model to directly generate a piano cover
from pop audio without melody and chord extraction modules.

Pop2Piano is an encoder-decoder Transformer model based on [T5](https://huggingface.co/papers/1910.10683). The input audio
is transformed to its waveform and passed to the encoder, which transforms it to a latent representation. The decoder
uses these latent representations to generate token ids in an autoregressive way. Each token id corresponds to one of four
different token types: time, velocity, note and 'special'. The token ids are then decoded to their equivalent MIDI file.

The abstract from the paper is the following:

*Piano covers of pop music are enjoyed by many people. However, the
task of automatically generating piano covers of pop music is still
understudied. This is partly due to the lack of synchronized
{Pop, Piano Cover} data pairs, which made it challenging to apply
the latest data-intensive deep learning-based methods. To leverage
the power of the data-driven approach, we make a large amount of
paired and synchronized {Pop, Piano Cover} data using an automated
pipeline. In this paper, we present Pop2Piano, a Transformer network
that generates piano covers given waveforms of pop music. To the best
of our knowledge, this is the first model to generate a piano cover
directly from pop audio without using melody and chord extraction
modules. We show that Pop2Piano, trained with our dataset, is capable
of producing plausible piano covers.*

This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/sweetcocoa/pop2piano).

## Usage tips

* To use Pop2Piano, you will need to install the 🤗 Transformers library, as well as the following third party modules:  

```bash
pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy
```

Please note that you may need to restart your runtime after installation.

* Pop2Piano is an Encoder-Decoder based model like T5.
* Pop2Piano can be used to generate midi-audio files for a given audio sequence.
* Choosing different composers in `Pop2PianoForConditionalGeneration.generate()` can lead to variety of different results.
* Setting the sampling rate to 44.1 kHz when loading the audio file can give good performance.
* Though Pop2Piano was mainly trained on Korean Pop music, it also does pretty well on other Western Pop or Hip Hop songs.

## Examples

- Example using HuggingFace Dataset:

```python
>>> from datasets import load_dataset
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

>>> inputs = processor(
...     audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
... )
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- Example using your own audio file:

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> audio, sr = librosa.load("", sr=44100)  # feel free to change the sr to a suitable value.
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- Example of processing multiple audio files in batch:

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> # feel free to change the sr to a suitable value.
>>> audio1, sr1 = librosa.load("", sr=44100)  
>>> audio2, sr2 = librosa.load("", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=[audio1, audio2], sampling_rate=[sr1, sr2], return_attention_mask=True, return_tensors="pt")
>>> # Since we now generating in batch(2 audios) we must pass the attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # Since we now have 2 generated MIDI files
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

- Example of processing multiple audio files in batch (Using `Pop2PianoFeatureExtractor` and `Pop2PianoTokenizer`):

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoFeatureExtractor, Pop2PianoTokenizer

>>> # feel free to change the sr to a suitable value.
>>> audio1, sr1 = librosa.load("", sr=44100)  
>>> audio2, sr2 = librosa.load("", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
>>> tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = feature_extractor(
...     audio=[audio1, audio2], 
...     sampling_rate=[sr1, sr2], 
...     return_attention_mask=True, 
...     return_tensors="pt",
... )
>>> # Since we now generating in batch(2 audios) we must pass the attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = tokenizer.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # Since we now have 2 generated MIDI files
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

## Pop2PianoConfig[[transformers.Pop2PianoConfig]]

#### transformers.Pop2PianoConfig[[transformers.Pop2PianoConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pop2piano/configuration_pop2piano.py#L24)

This is the configuration class to store the configuration of a [Pop2PianoForConditionalGeneration](/docs/transformers/main/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration). It is used
to instantiate a Pop2PianoForConditionalGeneration model according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
Pop2Piano [sweetcocoa/pop2piano](https://huggingface.co/sweetcocoa/pop2piano) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

vocab_size (`int`, *optional*, defaults to 2400) : Vocabulary size of the `Pop2PianoForConditionalGeneration` model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [Pop2PianoForConditionalGeneration](/docs/transformers/main/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration).

composer_vocab_size (`int`, *optional*, defaults to 21) : Denotes the number of composers.

d_model (`int`, *optional*, defaults to 512) : Size of the encoder layers and the pooler layer.

d_kv (`int`, *optional*, defaults to 64) : Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will be defined as `num_heads * d_kv`.

d_ff (`int`, *optional*, defaults to 2048) : Size of the intermediate feed forward layer in each `Pop2PianoBlock`.

num_layers (`int`, *optional*, defaults to 6) : Number of hidden layers in the Transformer encoder.

num_decoder_layers (`int`, *optional*) : Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.

num_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Transformer encoder.

relative_attention_num_buckets (`int`, *optional*, defaults to 32) : The number of buckets to use for each attention layer.

relative_attention_max_distance (`int`, *optional*, defaults to 128) : The maximum distance of the longer sequences for the bucket separation.

dropout_rate (`float`, *optional*, defaults to 0.1) : The ratio for all dropout layers.

layer_norm_epsilon (`float`, *optional*, defaults to 1e-6) : The epsilon used by the layer normalization layers.

initializer_factor (`float`, *optional*, defaults to 1.0) : A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization testing).

feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`) : Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models).

dense_act_fn (`string`, *optional*, defaults to `"relu"`) : Type of Activation Function to be used in `Pop2PianoDenseActDense` and in `Pop2PianoDenseGatedActDense`.

## Pop2PianoFeatureExtractor[[transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__..Placeholder]]

#### transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__..Placeholder[[transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__..Placeholder]]

__call__transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__..Placeholder.__call__[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]
Call self as a function.

## Pop2PianoForConditionalGeneration[[transformers.Pop2PianoForConditionalGeneration]]

#### transformers.Pop2PianoForConditionalGeneration[[transformers.Pop2PianoForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pop2piano/modeling_pop2piano.py#L949)

Pop2Piano Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Pop2PianoForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/pop2piano/modeling_pop2piano.py#L1037[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[tuple[tuple[torch.Tensor]]] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "input_features", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Pop2Piano is a model with relative position embeddings
  so you should be able to pad the inputs on both the right and the left. Indices can be obtained using
  [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.
  [What are input IDs?](../glossary#input-ids) To know more on how to prepare `input_ids` for pretraining
  take a look a [Pop2Piano Training](./Pop2Piano#training).
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
  [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.
  [What are decoder input IDs?](../glossary#decoder-input-ids) Pop2Piano uses the `pad_token_id` as the
  starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last
  `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
- **encoder_outputs** (`tuple[tuple[torch.Tensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **input_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) --
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Pop2PianoFeatureExtractor](/docs/transformers/main/en/model_doc/pop2piano#transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__..Placeholder). See `Pop2PianoFeatureExtractor.__call__()` for details ([Pop2PianoProcessor](/docs/transformers/main/en/model_doc/pop2piano#transformers.models.pop2piano.processing_pop2piano._LazyModule.__getattr__..Placeholder) uses
  [Pop2PianoFeatureExtractor](/docs/transformers/main/en/model_doc/pop2piano#transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__..Placeholder) for processing audios).
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
  config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.modeling_outputs.Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Pop2PianoConfig](/docs/transformers/main/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
The [Pop2PianoForConditionalGeneration](/docs/transformers/main/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoProcessor, Pop2PianoForConditionalGeneration
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
...

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
...
```

**Parameters:**

config ([Pop2PianoConfig](/docs/transformers/main/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Pop2PianoConfig](/docs/transformers/main/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
#### generate[[transformers.Pop2PianoForConditionalGeneration.generate]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pop2piano/modeling_pop2piano.py#L1153)

Generates token ids for midi outputs.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model's default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation
strategies and code examples, check out the [following guide](./generation_strategies).

**Parameters:**

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`.

attention_mask : For batched generation `input_features` are padded to have the same shape across all examples. `attention_mask` helps to determine which areas were padded and which were not. - 1 for tokens that are **not padded**, - 0 for tokens that are **padded**.

composer (`str`, *optional*, defaults to `"composer1"`) : This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each `"composer"`. Please make sure that the composer value is present in `composer_to_feature_token` in `generation_config`. For an example please see https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json .

generation_config (`~generation.GenerationConfig`, *optional*) : The generation configuration to be used as base parametrization for the generation call. `**kwargs` passed to generate matching the attributes of `generation_config` will override them. If `generation_config` is not provided, the default will be used, which had the following loading priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)'s default values, whose documentation should be checked to parameterize generation.

kwargs : Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

**Returns:**

`[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor``

A [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) types are:
- [GenerateEncoderDecoderOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
- [GenerateBeamEncoderDecoderOutput](/docs/transformers/main/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

## Pop2PianoTokenizer[[transformers.models.pop2piano.tokenization_pop2piano._LazyModule.__getattr__..Placeholder]]

#### transformers.models.pop2piano.tokenization_pop2piano._LazyModule.__getattr__..Placeholder[[transformers.models.pop2piano.tokenization_pop2piano._LazyModule.__getattr__..Placeholder]]

__call__transformers.models.pop2piano.tokenization_pop2piano._LazyModule.__getattr__..Placeholder.__call__[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]
Call self as a function.

## Pop2PianoProcessor[[transformers.models.pop2piano.processing_pop2piano._LazyModule.__getattr__..Placeholder]]

#### transformers.models.pop2piano.processing_pop2piano._LazyModule.__getattr__..Placeholder[[transformers.models.pop2piano.processing_pop2piano._LazyModule.__getattr__..Placeholder]]

__call__transformers.models.pop2piano.processing_pop2piano._LazyModule.__getattr__..Placeholder.__call__[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]
Call self as a function.
