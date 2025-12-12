*This model was released on 2021-11-15 and added to Hugging Face Transformers on 2021-11-30.*

# VisionTextDualEncoder

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The [VisionTextDualEncoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) can be used to initialize a vision-text dual encoder model with
any pretrained vision autoencoding model as the vision encoder (*e.g.* [ViT](vit), [BEiT](beit), [DeiT](deit)) and any pretrained text autoencoding model as the text encoder (*e.g.* [RoBERTa](roberta), [BERT](bert)). Two projection layers are added on top of both the vision and text encoder to project the output embeddings
to a shared latent space. The projection layers are randomly initialized so the model should be fine-tuned on a
downstream task. This model can be used to align the vision-text embeddings using CLIP like contrastive image-text
training and then can be used for zero-shot vision tasks such image-classification or retrieval.

In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://huggingface.co/papers/2111.07991) it is shown how
leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvement on
new zero-shot vision tasks such as image classification or retrieval.

## VisionTextDualEncoderConfig

### class transformers.VisionTextDualEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py#L34)

( projection\_dim = 512 logit\_scale\_init\_value = 2.6592 \*\*kwargs  )

Parameters

* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and vision projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original CLIP implementation.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig) is the configuration class to store the configuration of a
[VisionTextDualEncoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel). It is used to instantiate [VisionTextDualEncoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) model according to the
specified arguments, defining the text model and vision model configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

>>> # Initializing a BERT and ViT configuration
>>> config_vision = ViTConfig()
>>> config_text = BertConfig()

>>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

>>> # Initializing a BERT and ViT model (with random weights)
>>> model = VisionTextDualEncoderModel(config=config)

>>> # Accessing the model configuration
>>> config_vision = model.config.vision_config
>>> config_text = model.config.text_config

>>> # Saving the model, including its configuration
>>> model.save_pretrained("vit-bert")

>>> # loading model and config from pretrained folder
>>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
>>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)
```

#### from\_vision\_text\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py#L109)

( vision\_config: PretrainedConfig text\_config: PretrainedConfig \*\*kwargs  ) → [VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig)

Returns

[VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig)

An instance of a configuration object

Instantiate a [VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig) (or a derived class) from text model configuration and vision
model configuration.

## VisionTextDualEncoderProcessor

### class transformers.VisionTextDualEncoderProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py#L31)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** ([AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer), *optional*) —
  The tokenizer is a required input.

Constructs a VisionTextDualEncoder processor which wraps an image processor and a tokenizer into a single
processor.

[VisionTextDualEncoderProcessor](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderProcessor) offers all the functionalities of [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor) and [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer).
See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more
information.

## VisionTextDualEncoderModel

### class transformers.VisionTextDualEncoderModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L46)

( config: typing.Optional[transformers.models.vision\_text\_dual\_encoder.configuration\_vision\_text\_dual\_encoder.VisionTextDualEncoderConfig] = None vision\_model: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None text\_model: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None  )

Parameters

* **config** ([VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig), *optional*) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **vision\_model** (`~modeling_utils.PreTrainedModel`, *optional*) —
  The vision model to use.
* **text\_model** (`~modeling_utils.PreTrainedModel`, *optional*) —
  The text model to use.

The bare Vision Text Dual Encoder Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L187)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None return\_loss: typing.Optional[bool] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.clip.modeling_clip.CLIPOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([VisionTextDualEncoderProcessor](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderProcessor) uses
  `image_processor_class` for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **return\_loss** (`bool`, *optional*) —
  Whether or not to return the contrastive loss.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clip.modeling_clip.CLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clip.modeling_clip.CLIPOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [CLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [CLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPVisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [CLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [CLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPVisionModel).

The [VisionTextDualEncoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import (
...     VisionTextDualEncoderModel,
...     VisionTextDualEncoderProcessor,
...     AutoImageProcessor,
...     AutoTokenizer,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
>>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
...     "google/vit-base-patch16-224", "google-bert/bert-base-uncased"
... )

>>> # contrastive training
>>> urls = [
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
... ]
>>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
... )
>>> outputs = model(
...     input_ids=inputs.input_ids,
...     attention_mask=inputs.attention_mask,
...     pixel_values=inputs.pixel_values,
...     return_loss=True,
... )
>>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score

>>> # save and load from pretrained
>>> model.save_pretrained("vit-bert")
>>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert")

>>> # inference
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vision-text-dual-encoder.md)
