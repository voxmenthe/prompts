# Vision Encoder Decoder Models

## Overview

The [VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) can be used to initialize an image-to-text model with any
pretrained Transformer-based vision model as the encoder (*e.g.* [ViT](vit), [BEiT](beit), [DeiT](deit), [Swin](swin))
and any pretrained language model as the decoder (*e.g.* [RoBERTa](roberta), [GPT2](gpt2), [BERT](bert), [DistilBERT](distilbert)).

The effectiveness of initializing image-to-text-sequence models with pretrained checkpoints has been shown in (for
example) [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei.

After such a [VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples below
for more information).

An example application is image captioning, in which the encoder is used to encode the image, after which an autoregressive language model generates
the caption. Another example is optical character recognition. Refer to [TrOCR](trocr), which is an instance of [VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel).

## Randomly initializing `VisionEncoderDecoderModel` from model configurations

[VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [ViTModel](/docs/transformers/main/en/model_doc/vit#transformers.ViTModel) configuration for the encoder
and the default `BertForCausalLM` configuration for the decoder.

```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## Initialising `VisionEncoderDecoderModel` from a pretrained encoder and a pretrained decoder

[VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained Transformer-based vision model, *e.g.* [Swin](swin), can serve as the encoder and both pretrained auto-encoding models, *e.g.* BERT, pretrained causal language models, *e.g.* GPT2, as well as the pretrained decoder part of sequence-to-sequence models, *e.g.* decoder of BART, can be used as the decoder.
Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized.
Initializing [VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the *Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder).
To do so, the `VisionEncoderDecoderModel` class provides a [VisionEncoderDecoderModel.from_encoder_decoder_pretrained()](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained) method.

```python
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "google-bert/bert-base-uncased"
... )
```

## Loading an existing `VisionEncoderDecoderModel` checkpoint and perform inference

To load fine-tuned checkpoints of the `VisionEncoderDecoderModel` class, [VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the `generate` method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```python
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> # load a fine-tuned image captioning model and corresponding tokenizer and image processor
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> # let's perform inference on an image
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> # autoregressively generate caption (uses greedy decoding by default)
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
a cat laying on a blanket next to a cat laying on a bed
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model on a dataset of (image, text) pairs.
As you can see, only 2 inputs are required for the model in order to compute a loss: `pixel_values` (which are the
images) and `labels` (which are the `input_ids` of the encoded target sequence).

```python
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

This model was contributed by [nielsr](https://github.com/nielsrogge).

## VisionEncoderDecoderConfig[[transformers.VisionEncoderDecoderConfig]]

#### transformers.VisionEncoderDecoderConfig[[transformers.VisionEncoderDecoderConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py#L25)

[VisionEncoderDecoderConfig](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) is the configuration class to store the configuration of a
[VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel). It is used to instantiate a Vision-Encoder-Text-Decoder model according to the
specified arguments, defining the encoder and decoder configs.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> # Initializing a ViT & BERT style configuration
>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

>>> # Initializing a ViTBert model (with random weights) from a ViT & google-bert/bert-base-uncased style configurations
>>> model = VisionEncoderDecoderModel(config=config)

>>> # Accessing the model configuration
>>> config_encoder = model.config.encoder
>>> config_decoder = model.config.decoder
>>> # set decoder config to causal lm
>>> config_decoder.is_decoder = True
>>> config_decoder.add_cross_attention = True

>>> # Saving the model, including its configuration
>>> model.save_pretrained("my-model")

>>> # loading model and config from pretrained folder
>>> encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
>>> model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
```

from_encoder_decoder_configstransformers.VisionEncoderDecoderConfig.from_encoder_decoder_configshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py#L93[{"name": "encoder_config", "val": ": PreTrainedConfig"}, {"name": "decoder_config", "val": ": PreTrainedConfig"}, {"name": "**kwargs", "val": ""}][VisionEncoderDecoderConfig](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)An instance of a configuration object

Instantiate a [VisionEncoderDecoderConfig](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) (or a derived class) from a pre-trained encoder model
configuration and decoder model configuration.

**Parameters:**

kwargs (*optional*) : Dictionary of keyword arguments. Notably: 

encoder ([PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig), *optional*) : An instance of a configuration object that defines the encoder config.

decoder ([PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig), *optional*) : An instance of a configuration object that defines the decoder config.

**Returns:**

`[VisionEncoderDecoderConfig](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)`

An instance of a configuration object

## VisionEncoderDecoderModel[[transformers.VisionEncoderDecoderModel]]

#### transformers.VisionEncoderDecoderModel[[transformers.VisionEncoderDecoderModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L57)

The bare Vision Encoder Decoder Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.VisionEncoderDecoderModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L302[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[tuple[torch.FloatTensor]] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

  If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).

  For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
  right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
- **encoder_outputs** (`tuple[torch.FloatTensor]`, *optional*) --
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
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
  into associated vectors than the model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
  ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
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
elements depending on the configuration ([VisionEncoderDecoderConfig](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) and inputs.

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
The [VisionEncoderDecoderModel](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoProcessor, VisionEncoderDecoderModel
>>> import requests
>>> from PIL import Image
>>> import torch

>>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
>>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

>>> # load image from the IAM dataset
>>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> # training
>>> model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
>>> model.config.pad_token_id = processor.tokenizer.pad_token_id
>>> model.config.vocab_size = model.config.decoder.vocab_size

>>> pixel_values = processor(image, return_tensors="pt").pixel_values
>>> text = "hello world"
>>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
>>> outputs = model(pixel_values=pixel_values, labels=labels)
>>> loss = outputs.loss

>>> # inference (generation)
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig), *optional*) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

encoder (`PreTrainedModel`, *optional*) : The encoder model to use.

decoder (`PreTrainedModel`, *optional*) : The decoder model to use.

**Returns:**

`[transformers.modeling_outputs.Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.Seq2SeqLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisionEncoderDecoderConfig](/docs/transformers/main/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) and inputs.

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
#### from_encoder_decoder_pretrained[[transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L158)

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
checkpoints.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
the model, you need to first set it back in training mode with `model.train()`.

Example:

```python
>>> from transformers import VisionEncoderDecoderModel

>>> # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
... )
>>> # saving model after fine-tuning
>>> model.save_pretrained("./vit-bert")
>>> # load fine-tuned model
>>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")
```

**Parameters:**

encoder_pretrained_model_name_or_path (`str`, *optional*) : Information necessary to initiate the image encoder. Can be either:  - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An example is `google/vit-base-patch16-224-in21k`. - A path to a *directory* containing model weights saved using [save_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`. 

decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`) : Information necessary to initiate the text decoder. Can be either:  - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. - A path to a *directory* containing model weights saved using [save_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`. 

model_args (remaining positional arguments, *optional*) : All remaining positional arguments will be passed to the underlying model's `__init__` method. 

kwargs (remaining dictionary of keyword arguments, *optional*) : Can be used to update the configuration object (after it being loaded) and initiate the model (e.g., `output_attentions=True`).  - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter. - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter. - To update the parent model configuration, do not use a prefix for each configuration parameter.  Behaves differently depending on whether a `config` is provided or automatically loaded.
