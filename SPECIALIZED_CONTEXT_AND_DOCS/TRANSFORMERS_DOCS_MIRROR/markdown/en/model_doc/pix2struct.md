*This model was released on 2022-10-07 and added to Hugging Face Transformers on 2023-03-22.*

# Pix2Struct

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://huggingface.co/papers/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova.

The abstract from the paper is the following:

> Visually-situated language is ubiquitous — sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms. Perhaps due to this diversity, previous work has typically relied on domain-specific recipes with limited sharing of the underlying data, model architectures, and objectives. We present Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, image captioning. In addition to the novel pretraining strategy, we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions are rendered directly on top of the input image. For the first time, we show that a single pretrained model can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images.

Tips:

Pix2Struct has been fine tuned on a variety of tasks and datasets, ranging from image captioning, visual question answering (VQA) over different inputs (books, charts, science diagrams), captioning UI components etc. The full list can be found in Table 1 of the paper.
We therefore advise you to use these models for the tasks they have been fine tuned on. For instance, if you want to use Pix2Struct for UI captioning, you should use the model fine tuned on the UI dataset. If you want to use Pix2Struct for image captioning, you should use the model fine tuned on the natural images captioning dataset and so on.

If you want to use the model to perform conditional text captioning, make sure to use the processor with `add_special_tokens=False`.

This model was contributed by [ybelkada](https://huggingface.co/ybelkada).
The original code can be found [here](https://github.com/google-research/pix2struct).

## Resources

* [Fine-tuning Notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)
* [All models](https://huggingface.co/models?search=pix2struct)

## Pix2StructConfig

### class transformers.Pix2StructConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/configuration_pix2struct.py#L247)

( text\_config = None vision\_config = None initializer\_factor = 1.0 initializer\_range = 0.02 is\_vqa = False tie\_word\_embeddings = False is\_encoder\_decoder = True \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Pix2StructTextConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructTextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Pix2StructVisionConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructVisionConfig).
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  Factor to multiply the initialization range with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **is\_vqa** (`bool`, *optional*, defaults to `False`) —
  Whether the model has been fine-tuned for VQA or not.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig) is the configuration class to store the configuration of a
[Pix2StructForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration). It is used to instantiate a Pix2Struct model according to the specified
arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will
yield a similar configuration to that of the Pix2Struct-base
[google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

>>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
>>> configuration = Pix2StructConfig()

>>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
>>> model = Pix2StructForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

>>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
>>> config_text = Pix2StructTextConfig()
>>> config_vision = Pix2StructVisionConfig()

>>> config = Pix2StructConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/configuration_utils.py#L1254)

( text\_config vision\_config \*\*kwargs  ) → `PreTrainedConfig`

Returns

`PreTrainedConfig`

An instance of a configuration object

Instantiate a model config (or a derived class) from text model configuration and vision model
configuration.

## Pix2StructTextConfig

### class transformers.Pix2StructTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/configuration_pix2struct.py#L24)

( vocab\_size = 50244 hidden\_size = 768 d\_kv = 64 d\_ff = 2048 num\_layers = 12 num\_heads = 12 relative\_attention\_num\_buckets = 32 relative\_attention\_max\_distance = 128 dropout\_rate = 0.1 layer\_norm\_epsilon = 1e-06 initializer\_factor = 1.0 dense\_act\_fn = 'gelu\_new' decoder\_start\_token\_id = 0 use\_cache = False pad\_token\_id = 0 eos\_token\_id = 1 tie\_word\_embeddings = False is\_decoder = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 50244) —
  Vocabulary size of the `Pix2Struct` text model. Defines the number of different tokens that can be
  represented by the `inputs_ids` passed when calling [Pix2StructTextModel](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructTextModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **d\_kv** (`int`, *optional*, defaults to 64) —
  Dimensionality of the key, query, value projections in each attention head.
* **d\_ff** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **relative\_attention\_num\_buckets** (`int`, *optional*, defaults to 32) —
  The number of buckets to use for each attention layer.
* **relative\_attention\_max\_distance** (`int`, *optional*, defaults to 128) —
  The maximum distance of the longer sequences for the bucket separation.
* **dropout\_rate** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-6) —
  The epsilon used by the layer normalization layers.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).
* **dense\_act\_fn** (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`) —
  The non-linear activation function (function or string).
* **decoder\_start\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the `decoder_start_token_id` token.
* **use\_cache** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the `padding` token.
* **eos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the `end-of-sequence` token.

This is the configuration class to store the configuration of a [Pix2StructTextModel](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructTextModel). It is used to instantiate
a Pix2Struct text model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Pix2Struct text decoder used by
the [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Pix2StructTextConfig, Pix2StructTextModel

>>> # Initializing a Pix2StructTextConfig with google/pix2struct-base style configuration
>>> configuration = Pix2StructTextConfig()

>>> # Initializing a Pix2StructTextModel (with random weights) from the google/pix2struct-base style configuration
>>> model = Pix2StructTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Pix2StructVisionConfig

### class transformers.Pix2StructVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/configuration_pix2struct.py#L148)

( hidden\_size = 768 patch\_embed\_hidden\_size = 768 d\_ff = 2048 d\_kv = 64 num\_hidden\_layers = 12 num\_attention\_heads = 12 dense\_act\_fn = 'gelu\_new' layer\_norm\_eps = 1e-06 dropout\_rate = 0.0 attention\_dropout = 0.0 initializer\_range = 1e-10 initializer\_factor = 1.0 seq\_len = 4096 relative\_attention\_num\_buckets = 32 relative\_attention\_max\_distance = 128 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **patch\_embed\_hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the input patch\_embedding layer in the Transformer encoder.
* **d\_ff** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **d\_kv** (`int`, *optional*, defaults to 64) —
  Dimensionality of the key, query, value projections per attention head.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **dense\_act\_fn** (`str` or `function`, *optional*, defaults to `"gelu_new"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **dropout\_rate** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 1e-10) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).
* **seq\_len** (`int`, *optional*, defaults to 4096) —
  Maximum sequence length (here number of patches) supported by the model.
* **relative\_attention\_num\_buckets** (`int`, *optional*, defaults to 32) —
  The number of buckets to use for each attention layer.
* **relative\_attention\_max\_distance** (`int`, *optional*, defaults to 128) —
  The maximum distance (in tokens) to use for each attention layer.

This is the configuration class to store the configuration of a [Pix2StructVisionModel](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructVisionModel). It is used to
instantiate a Pix2Struct vision model according to the specified arguments, defining the model architecture.
Instantiating a configuration defaults will yield a similar configuration to that of the Pix2Struct-base
[google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

>>> # Initializing a Pix2StructVisionConfig with google/pix2struct-base style configuration
>>> configuration = Pix2StructVisionConfig()

>>> # Initializing a Pix2StructVisionModel (with random weights) from the google/pix2struct-base style configuration
>>> model = Pix2StructVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Pix2StructProcessor

### class transformers.Pix2StructProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/processing_pix2struct.py#L55)

( image\_processor tokenizer  )

Parameters

* **image\_processor** (`Pix2StructImageProcessor`) —
  An instance of [Pix2StructImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructImageProcessor). The image processor is a required input.
* **tokenizer** (Union[`T5TokenizerFast`, `T5Tokenizer`]) —
  An instance of [‘T5TokenizerFast`] or [‘T5Tokenizer`]. The tokenizer is a required input.

Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single
processor.

[Pix2StructProcessor](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructProcessor) offers all the functionalities of [Pix2StructImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructImageProcessor) and [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast). See
the docstring of `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Pix2StructImageProcessor

### class transformers.Pix2StructImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/image_processing_pix2struct.py#L189)

( do\_convert\_rgb: bool = True do\_normalize: bool = True patch\_size: typing.Optional[dict[str, int]] = None max\_patches: int = 2048 is\_vqa: bool = False \*\*kwargs  )

Parameters

* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
  deviation.
* **patch\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 16, "width": 16}`):
  The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
* **max\_patches** (`int`, *optional*, defaults to 2048) —
  The maximum number of patches to extract from the image as per the [Pix2Struct
  paper](https://huggingface.co/papers/2210.03347).
* **is\_vqa** (`bool`, *optional*, defaults to `False`) —
  Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is
  rendered onto the input images.

Constructs a Pix2Struct image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/image_processing_pix2struct.py#L348)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] header\_text: typing.Optional[str] = None do\_convert\_rgb: typing.Optional[bool] = None do\_normalize: typing.Optional[bool] = None max\_patches: typing.Optional[int] = None patch\_size: typing.Optional[dict[str, int]] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images.
* **header\_text** (`Union[list[str], str]`, *optional*) —
  Text to render as a header. Only has an effect if `image_processor.is_vqa` is `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **max\_patches** (`int`, *optional*, defaults to `self.max_patches`) —
  Maximum number of patches to extract.
* **patch\_size** (`dict`, *optional*, defaults to `self.patch_size`) —
  Dictionary containing the patch height and width.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images. The processor first computes the maximum possible number of
aspect-ratio preserving patches of size `patch_size` that can be extracted from the image. It then pads the
image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the
images are standardized following the tensorflow implementation of `per_image_standardization`
(<https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization>).

## Pix2StructTextModel

### class transformers.Pix2StructTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/modeling_pix2struct.py#L1020)

( config  )

Parameters

* **config** ([Pix2StructTextModel](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructTextModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone text decoder of Pix2Struct

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/modeling_pix2struct.py#L1048)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Pix2StructText is a model with relative position
  embeddings so you should be able to pad the inputs on both the right and the left.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail.

  [What are input IDs?](../glossary#input-ids)

  To know more on how to prepare `input_ids` for pretraining take a look a [Pix2StructText
  Training](./t5#training).
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **inputs\_embeds** (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
  `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.

The [Pix2StructTextModel](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, Pix2StructTextModel

>>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
>>> model = Pix2StructTextModel.from_pretrained("google/pix2struct-textcaps-base")

>>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)
>>> loss = outputs.loss
```

## Pix2StructVisionModel

### class transformers.Pix2StructVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/modeling_pix2struct.py#L477)

( config: Pix2StructVisionConfig  )

Parameters

* **config** ([Pix2StructVisionConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pix2Struct Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/modeling_pix2struct.py#L506)

( flattened\_patches: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **flattened\_patches** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels x patch_height x patch_width)`) —
  Flattened and padded pixel values. These values can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See
  `Pix2StructVisionImageProcessor.__call__` for details. Check the [original
  paper](https://huggingface.co/papers/2210.03347) (figure 5) for more details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Pix2StructVisionModel](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import requests
>>> from PIL import Image
>>> from transformers import AutoProcessor, Pix2StructVisionModel

>>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
>>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 2048, 768]
```

## Pix2StructForConditionalGeneration

### class transformers.Pix2StructForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/modeling_pix2struct.py#L1396)

( config: Pix2StructConfig  )

Parameters

* **config** ([Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

A conditional generation model with a language modeling head. Can be used for sequence generation tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pix2struct/modeling_pix2struct.py#L1427)

( flattened\_patches: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None labels: typing.Optional[torch.LongTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **flattened\_patches** (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`) —
  Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =
  `num_channels`  *`patch_size`*  `patch_size`

  The process of flattening the pixel patches is done by `Pix2StructProcessor`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).

  To know more on how to prepare `decoder_input_ids` for pretraining take a look at [Pix2StructText
  Training](./t5#training).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
  `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss for the decoder.
* **decoder\_inputs\_embeds** (`torch.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [Pix2StructForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

Inference:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
>>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> # autoregressive generation
>>> generated_ids = model.generate(**inputs, max_new_tokens=50)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
A stop sign is on a street corner.

>>> # conditional generation
>>> text = "A picture of"
>>> inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)

>>> generated_ids = model.generate(**inputs, max_new_tokens=50)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
A picture of a stop sign with a red stop sign
```

Training:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("google/pix2struct-base")
>>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "A stop sign is on the street corner."

>>> inputs = processor(images=image, return_tensors="pt")
>>> labels = processor(text=text, return_tensors="pt").input_ids

>>> # forward pass
>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> print(f"{loss.item():.5f}")
5.94282
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/pix2struct.md)
