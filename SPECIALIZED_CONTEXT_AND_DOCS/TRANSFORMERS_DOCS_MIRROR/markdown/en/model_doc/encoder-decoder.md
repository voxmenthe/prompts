*This model was released on 2017-06-12 and added to Hugging Face Transformers on 2020-11-16.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Encoder Decoder Models

[`EncoderDecoderModel`](https://huggingface.co/papers/1706.03762) initializes a sequence-to-sequence model with any pretrained autoencoder and pretrained autoregressive model. It is effective for sequence generation tasks as demonstrated in [Text Summarization with Pretrained Encoders](https://huggingface.co/papers/1908.08345) which uses [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel) as the encoder and decoder.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf).

Click on the Encoder Decoder models in the right sidebar for more examples of how to apply Encoder Decoder to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="patrickvonplaten/bert2bert-cnn_dailymail-fp16",
    device=0
)

text = "Plants create energy through a process known as photosynthesis. This involves capturing sunlight and converting carbon dioxide and water into glucose and oxygen."
print(summarizer(text))
```

## Notes

* [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) can be initialized using any pretrained encoder and decoder. But depending on the decoder architecture, the cross-attention layers may be randomly initialized.

These models require downstream fine-tuning, as discussed in this [blog post](https://huggingface.co/blog/warm-starting-encoder-decoder). Use [from\_encoder\_decoder\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel.from_encoder_decoder_pretrained) to combine encoder and decoder checkpoints.


```
from transformers import EncoderDecoderModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-uncased"
)
```

* Encoder Decoder models can be fine-tuned like BART, T5 or any other encoder-decoder model. Only 2 inputs are required to compute a loss, `input_ids` and `labels`. Refer to this [notebook](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) for a more detailed training example.


```
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

* [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) can be randomly initialized from an encoder and a decoder config as shown below.


```
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

* The Encoder Decoder Model can also be used for translation as shown below.


```
from transformers import AutoTokenizer, EncoderDecoderModel

# Load a pre-trained translation model
model_name = "google/bert2bert_L-24_wmt_en_de"
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = EncoderDecoderModel.from_pretrained(model_name)

# Input sentence to translate
input_text = "Plants create energy through a process known as"

# Encode the input text
inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids

# Generate the translated output
outputs = model.generate(inputs)[0]

# Decode the output tokens to get the translated sentence
translated_text = tokenizer.decode(outputs, skip_special_tokens=True)

print("Translated text:", translated_text)
```

## EncoderDecoderConfig

### class transformers.EncoderDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/encoder_decoder/configuration_encoder_decoder.py#L26)

( \*\*kwargs  )

Parameters

* **kwargs** (*optional*) —
  Dictionary of keyword arguments. Notably:
  + **encoder** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that defines
    the encoder config.
  + **decoder** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that defines
    the decoder config.

[EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) is the configuration class to store the configuration of a [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel). It is
used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder
configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> # Initializing a BERT google-bert/bert-base-uncased style configuration
>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

>>> # Initializing a Bert2Bert model (with random weights) from the google-bert/bert-base-uncased style configurations
>>> model = EncoderDecoderModel(config=config)

>>> # Accessing the model configuration
>>> config_encoder = model.config.encoder
>>> config_decoder = model.config.decoder
>>> # set decoder config to causal lm
>>> config_decoder.is_decoder = True
>>> config_decoder.add_cross_attention = True

>>> # Saving the model, including its configuration
>>> model.save_pretrained("my-model")

>>> # loading model and config from pretrained folder
>>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
>>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
```

#### from\_encoder\_decoder\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/encoder_decoder/configuration_encoder_decoder.py#L93)

( encoder\_config: PretrainedConfig decoder\_config: PretrainedConfig \*\*kwargs  ) → [EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)

Returns

[EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)

An instance of a configuration object

Instantiate a [EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) (or a derived class) from a pre-trained encoder model configuration and
decoder model configuration.

## EncoderDecoderModel

### class transformers.EncoderDecoderModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L68)

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

The bare Encoder Decoder Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L445)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None encoder\_outputs: typing.Optional[tuple[torch.FloatTensor]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
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
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)) and inputs.

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

The [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import EncoderDecoderModel, BertTokenizer
>>> import torch

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google-bert/bert-base-uncased", "google-bert/bert-base-uncased"
... )  # initialize Bert2Bert from pre-trained checkpoints

>>> # training
>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id
>>> model.config.vocab_size = model.config.decoder.vocab_size

>>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
>>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
>>> outputs = model(input_ids=input_ids, labels=labels)
>>> loss, logits = outputs.loss, outputs.logits

>>> # save and load from pretrained
>>> model.save_pretrained("bert2bert")
>>> model = EncoderDecoderModel.from_pretrained("bert2bert")

>>> # generation
>>> generated = model.generate(input_ids)
```

#### from\_encoder\_decoder\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L299)

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
>>> from transformers import EncoderDecoderModel

>>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")
>>> # saving model after fine-tuning
>>> model.save_pretrained("./bert2bert")
>>> # load fine-tuned model
>>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/encoder-decoder.md)
