*This model was released on 2023-06-26 and added to Hugging Face Transformers on 2023-10-30.*

# KOSMOS-2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The KOSMOS-2 model was proposed in [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://huggingface.co/papers/2306.14824) by Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei.

KOSMOS-2 is a Transformer-based causal language model and is trained using the next-word prediction task on a web-scale
dataset of grounded image-text pairs [GRIT](https://huggingface.co/datasets/zzliang/GRIT). The spatial coordinates of
the bounding boxes in the dataset are converted to a sequence of location tokens, which are appended to their respective
entity text spans (for example, `a snowman` followed by `<patch_index_0044><patch_index_0863>`). The data format is
similar to “hyperlinks” that connect the object regions in an image to their text span in the corresponding caption.

The abstract from the paper is the following:

*We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., “[text span](bounding boxes)”, where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GrIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability into downstream applications. We evaluate Kosmos-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at <https://aka.ms/kosmos-2>.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos_2_overview.jpg) Overview of tasks that KOSMOS-2 can handle. Taken from the [original paper](https://huggingface.co/papers/2306.14824).

## Example


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

>>> model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
>>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

>>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "<grounding> An image of"

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")

>>> generated_ids = model.generate(
...     pixel_values=inputs["pixel_values"],
...     input_ids=inputs["input_ids"],
...     attention_mask=inputs["attention_mask"],
...     image_embeds=None,
...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
...     use_cache=True,
...     max_new_tokens=64,
... )
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
>>> processed_text
'<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.'

>>> caption, entities = processor.post_process_generation(generated_text)
>>> caption
'An image of a snowman warming himself by a fire.'

>>> entities
[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
```

This model was contributed by [Yih-Dar SHIEH](https://huggingface.co/ydshieh). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/kosmos-2).

## Kosmos2Config

### class transformers.Kosmos2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/configuration_kosmos2.py#L206)

( text\_config = None vision\_config = None latent\_query\_num = 64 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize `Kosmos2TextConfig`.
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize `Kosmos2VisionConfig`.
* **latent\_query\_num** (`int`, *optional*, defaults to 64) —
  The number of latent query tokens that represent the image features used in the text decoder component.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

This is the configuration class to store the configuration of a [Kosmos2Model](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Model). It is used to instantiate a
KOSMOS-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the KOSMOS-2
[microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

Example:


```
>>> from transformers import Kosmos2Config, Kosmos2Model

>>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
>>> configuration = Kosmos2Config()

>>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
>>> model = Kosmos2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Kosmos2ImageProcessor

## Kosmos2Processor

### class transformers.Kosmos2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/processing_kosmos2.py#L68)

( image\_processor tokenizer num\_patch\_index\_tokens = 1024 \*kwargs  )

Parameters

* **image\_processor** (`CLIPImageProcessor`) —
  An instance of [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). The image processor is a required input.
* **tokenizer** (`XLMRobertaTokenizerFast`) —
  An instance of [‘XLMRobertaTokenizerFast`]. The tokenizer is a required input.
* **num\_patch\_index\_tokens** (`int`, *optional*, defaults to 1024) —
  The number of tokens that represent patch indices.

Constructs an KOSMOS-2 processor which wraps a KOSMOS-2 image processor and a KOSMOS-2 tokenizer into a single
processor.

[Kosmos2Processor](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Processor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) and some functionalities of
[XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast). See the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Processor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode)
for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/processing_kosmos2.py#L135)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None text: typing.Union[str, list[str]] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.kosmos2.processing\_kosmos2.Kosmos2ProcessorKwargs]  )

Parameters

* **bboxes** (`Union[list[tuple[int]], list[tuple[float]], list[list[tuple[int]]], list[list[tuple[float]]]]`, *optional*) —
  The bounding bboxes associated to `texts`.
* **num\_image\_tokens** (`int`, *optional* defaults to 64) —
  The number of (consecutive) places that are used to mark the placeholders to store image information.
  This should be the same as `latent_query_num` in the instance of `Kosmos2Config` you are using.
* **first\_image\_token\_id** (`int`, *optional*) —
  The token id that will be used for the first place of the subsequence that is reserved to store image
  information. If unset, will default to `self.tokenizer.unk_token_id + 1`.
* **add\_eos\_token** (`bool`, defaults to `False`) —
  Whether or not to include `EOS` token id in the encoding when `add_special_tokens=True`.

This method uses [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) method to prepare image(s) for the model, and
[XLMRobertaTokenizerFast.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) to prepare text for the model.

Please refer to the docstring of the above two methods for more information.

The rest of this documentation shows the arguments specific to `Kosmos2Processor`.

## Kosmos2Model

### class transformers.Kosmos2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/modeling_kosmos2.py#L1515)

( config: Kosmos2Config  )

Parameters

* **config** ([Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

KOSMOS-2 Model for generating text and image features. The model consists of a vision encoder and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/modeling_kosmos2.py#L1566)

( pixel\_values: typing.Optional[torch.Tensor] = None input\_ids: typing.Optional[torch.Tensor] = None image\_embeds\_position\_mask: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None image\_embeds: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.kosmos2.modeling_kosmos2.Kosmos2ModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **image\_embeds\_position\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the location in a sequence to insert the image features . Mask values selected in `[0, 1]`:
  + 1 for places where to put the image features,
  + 0 for places that are not for image features (i.e. for text tokens).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.kosmos2.modeling_kosmos2.Kosmos2ModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.kosmos2.modeling_kosmos2.Kosmos2ModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*) — Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **projection\_attentions** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
  the weighted average in the self-attention heads.
* **vision\_model\_output** (`BaseModelOutputWithPooling`, *optional*) — The output of the `Kosmos2VisionModel`.

The [Kosmos2Model](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Kosmos2Model

>>> model = Kosmos2Model.from_pretrained("microsoft/kosmos-2-patch14-224")
>>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

>>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> text = (
...     "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863>"
...     "</object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911>"
...     "</object>"
... )

>>> inputs = processor(text=text, images=image, return_tensors="pt", add_eos_token=True)

>>> last_hidden_state = model(
...     pixel_values=inputs["pixel_values"],
...     input_ids=inputs["input_ids"],
...     attention_mask=inputs["attention_mask"],
...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
... ).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 91, 2048]
```

## Kosmos2ForConditionalGeneration

### class transformers.Kosmos2ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/modeling_kosmos2.py#L1674)

( config: Kosmos2Config  )

Parameters

* **config** ([Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

KOSMOS-2 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a
language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kosmos2/modeling_kosmos2.py#L1702)

( pixel\_values: typing.Optional[torch.Tensor] = None input\_ids: typing.Optional[torch.Tensor] = None image\_embeds\_position\_mask: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None image\_embeds: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.kosmos2.modeling_kosmos2.Kosmos2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **image\_embeds\_position\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the location in a sequence to insert the image features . Mask values selected in `[0, 1]`:
  + 1 for places where to put the image features,
  + 0 for places that are not for image features (i.e. for text tokens).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.kosmos2.modeling_kosmos2.Kosmos2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.kosmos2.modeling_kosmos2.Kosmos2ForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, latent_query_num, hidden_size)`, *optional*) — Sequence of hidden-states at the output of `Kosmos2ImageToTextProjection`.
* **projection\_attentions** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights given by `Kosmos2ImageToTextProjection`, after the attention softmax, used to compute
  the weighted average in the self-attention heads.
* **vision\_model\_output** (`BaseModelOutputWithPooling`, *optional*) — The output of the `Kosmos2VisionModel`.

The [Kosmos2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

>>> model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
>>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

>>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "<grounding> An image of"

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")

>>> generated_ids = model.generate(
...     pixel_values=inputs["pixel_values"],
...     input_ids=inputs["input_ids"],
...     attention_mask=inputs["attention_mask"],
...     image_embeds=None,
...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
...     use_cache=True,
...     max_new_tokens=64,
... )
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
>>> processed_text
'<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.'

>>> caption, entities = processor.post_process_generation(generated_text)
>>> caption
'An image of a snowman warming himself by a fire.'

>>> entities
[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/kosmos-2.md)
