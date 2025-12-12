*This model was released on 2021-04-14 and added to Hugging Face Transformers on 2021-09-01.*

# Speech Encoder Decoder Models

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

The [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) can be used to initialize a speech-to-text model
with any pretrained speech autoencoding model as the encoder (*e.g.* [Wav2Vec2](wav2vec2), [Hubert](hubert)) and any pretrained autoregressive model as the decoder.

The effectiveness of initializing speech-sequence-to-text-sequence models with pretrained checkpoints for speech
recognition and speech translation has *e.g.* been shown in [Large-Scale Self- and Semi-Supervised Learning for Speech
Translation](https://huggingface.co/papers/2104.06678) by Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli,
Alexis Conneau.

An example of how to use a [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) for inference can be seen in [Speech2Text2](speech_to_text_2).

## Randomly initializing SpeechEncoderDecoderModel from model configurations.

[SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) configuration for the encoder
and the default `BertForCausalLM` configuration for the decoder.


```
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = SpeechEncoderDecoderModel(config=config)
```

## Initialising SpeechEncoderDecoderModel from a pretrained encoder and a pretrained decoder.

[SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained Transformer-based speech model, *e.g.* [Wav2Vec2](wav2vec2), [Hubert](hubert) can serve as the encoder and both pretrained auto-encoding models, *e.g.* BERT, pretrained causal language models, *e.g.* GPT2, as well as the pretrained decoder part of sequence-to-sequence models, *e.g.* decoder of BART, can be used as the decoder.
Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized.
Initializing [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the *Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder).
To do so, the `SpeechEncoderDecoderModel` class provides a [SpeechEncoderDecoderModel.from\_encoder\_decoder\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel.from_encoder_decoder_pretrained) method.


```
>>> from transformers import SpeechEncoderDecoderModel

>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/hubert-large-ll60k", "google-bert/bert-base-uncased"
... )
```

## Loading an existing SpeechEncoderDecoderModel checkpoint and perform inference.

To load fine-tuned checkpoints of the `SpeechEncoderDecoderModel` class, [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the `generate` method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.


```
>>> from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> # load a fine-tuned speech translation model and corresponding processor
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> # let's perform inference on a piece of English speech (which we'll translate to German)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # autoregressively generate transcription (uses greedy decoding by default)
>>> generated_ids = model.generate(input_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model on a dataset of (speech, text) pairs.
As you can see, only 2 inputs are required for the model in order to compute a loss: `input_values` (which are the
speech inputs) and `labels` (which are the `input_ids` of the encoded target sequence).


```
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset

>>> encoder_id = "facebook/wav2vec2-base-960h"  # acoustic model encoder
>>> decoder_id = "google-bert/bert-base-uncased"  # text decoder

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
>>> tokenizer = AutoTokenizer.from_pretrained(decoder_id)
>>> # Combine pre-trained encoder and pre-trained decoder to form a Seq2Seq model
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> # load an audio input and pre-process (normalise mean/std to 0/1)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # load its corresponding transcription and tokenize to generate labels
>>> labels = tokenizer(ds[0]["text"], return_tensors="pt").input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_values=input_values, labels=labels).loss
>>> loss.backward()
```

## SpeechEncoderDecoderConfig

### class transformers.SpeechEncoderDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py#L26)

( \*\*kwargs  )

Parameters

* **kwargs** (*optional*) —
  Dictionary of keyword arguments. Notably:
  + **encoder** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that defines
    the encoder config.
  + **decoder** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that defines
    the decoder config.

[SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) is the configuration class to store the configuration of a
[SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel). It is used to instantiate an Encoder Decoder model according to the specified
arguments, defining the encoder and decoder configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> # Initializing a Wav2Vec2 & BERT style configuration
>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

>>> # Initializing a Wav2Vec2Bert model from a Wav2Vec2 & google-bert/bert-base-uncased style configurations
>>> model = SpeechEncoderDecoderModel(config=config)

>>> # Accessing the model configuration
>>> config_encoder = model.config.encoder
>>> config_decoder = model.config.decoder
>>> # set decoder config to causal lm
>>> config_decoder.is_decoder = True
>>> config_decoder.add_cross_attention = True

>>> # Saving the model, including its configuration
>>> model.save_pretrained("my-model")

>>> # loading model and config from pretrained folder
>>> encoder_decoder_config = SpeechEncoderDecoderConfig.from_pretrained("my-model")
>>> model = SpeechEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
```

#### from\_encoder\_decoder\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py#L94)

( encoder\_config: PretrainedConfig decoder\_config: PretrainedConfig \*\*kwargs  ) → [SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig)

Returns

[SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig)

An instance of a configuration object

Instantiate a [SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) (or a derived class) from a pre-trained encoder model
configuration and decoder model configuration.

## SpeechEncoderDecoderModel

### class transformers.SpeechEncoderDecoderModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py#L56)

( config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None encoder: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None decoder: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **encoder** (`PreTrainedModel`, *optional*) —
  The encoder model to use.
* **decoder** (`PreTrainedModel`, *optional*) —
  The decoder model to use.

The bare Speech Encoder Decoder Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py#L317)

( inputs: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None encoder\_outputs: typing.Optional[tuple[torch.FloatTensor]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None input\_values: typing.Optional[torch.FloatTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`, *optional*) —
  Float values of input raw speech waveform or speech features. Values can be obtained by loading a `.flac`
  or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.*
  via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `inputs`, either the [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) or
  [Speech2TextProcessor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) should be used for padding and conversion into a tensor of type
  `torch.FloatTensor`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

  If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).

  For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
  right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **encoder\_outputs** (`tuple[torch.FloatTensor]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
  into associated vectors than the model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
  into an array of type *list[float]* or a *numpy.ndarray*, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into *input\_values*, the [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) should be used for padding and conversion
  into a tensor of type *torch.FloatTensor*. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  `feature_extractor_class`. See `feature_extractor_class.__call__` for details (`processor_class` uses
  `feature_extractor_class` for processing audios).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import SpeechEncoderDecoderModel, AutoProcessor
>>> from datasets import load_dataset
>>> import torch

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
>>> # Inference: Translate English speech to German
>>> generated = model.generate(input_values)
>>> decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
>>> decoded
'Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.'

>>> # Training: Train model on English transcription
>>> labels = processor(text=ds[0]["text"], return_tensors="pt").input_ids

>>> loss = model(input_values, labels=labels).loss
>>> loss.backward()
```

#### from\_encoder\_decoder\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py#L166)

( encoder\_pretrained\_model\_name\_or\_path: typing.Optional[str] = None decoder\_pretrained\_model\_name\_or\_path: typing.Optional[str] = None \*model\_args \*\*kwargs  )

Parameters

* **encoder\_pretrained\_model\_name\_or\_path** (`str`, *optional*) —
  Information necessary to initiate the encoder. Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **decoder\_pretrained\_model\_name\_or\_path** (`str`, *optional*, defaults to `None`) —
  Information necessary to initiate the decoder. Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (remaining positional arguments, *optional*) —
  All remaining positional arguments will be passed to the underlying model’s `__init__` method.
* **kwargs** (remaining dictionary of keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`).
  + To update the encoder configuration, use the prefix *encoder\_* for each configuration parameter.
  + To update the decoder configuration, use the prefix *decoder\_* for each configuration parameter.
  + To update the parent model configuration, do not use a prefix for each configuration parameter.

  Behaves differently depending on whether a `config` is provided or automatically loaded.

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
checkpoints.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
the model, you need to first set it back in training mode with `model.train()`.

Example:


```
>>> from transformers import SpeechEncoderDecoderModel

>>> # initialize a wav2vec2bert from a pretrained Wav2Vec2 and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/wav2vec2-base-960h", "google-bert/bert-base-uncased"
... )
>>> # saving model after fine-tuning
>>> model.save_pretrained("./wav2vec2bert")
>>> # load fine-tuned model
>>> model = SpeechEncoderDecoderModel.from_pretrained("./wav2vec2bert")
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/speech-encoder-decoder.md)
