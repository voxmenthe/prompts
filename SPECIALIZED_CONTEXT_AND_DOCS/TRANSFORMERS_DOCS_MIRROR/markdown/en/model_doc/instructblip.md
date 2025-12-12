*This model was released on 2023-05-11 and added to Hugging Face Transformers on 2023-06-26.*

# InstructBLIP

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The InstructBLIP model was proposed in [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://huggingface.co/papers/2305.06500) by Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.
InstructBLIP leverages the [BLIP-2](blip2) architecture for visual instruction tuning.

The abstract from the paper is the following:

*General-purpose language models that can solve various language-domain tasks have emerged driven by the pre-training and instruction-tuning pipeline. However, building general-purpose vision-language models is challenging due to the increased task discrepancy introduced by the additional visual input. Although vision-language pre-training has been widely studied, vision-language instruction tuning remains relatively less explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pre-trained BLIP-2 models. We gather a wide variety of 26 publicly available datasets, transform them into instruction tuning format and categorize them into two clusters for held-in instruction tuning and held-out zero-shot evaluation. Additionally, we introduce instruction-aware visual feature extraction, a crucial method that enables the model to extract informative features tailored to the given instruction. The resulting InstructBLIP models achieve state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and the larger Flamingo. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA IMG). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg) InstructBLIP architecture. Taken from the [original paper.](https://huggingface.co/papers/2305.06500)

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

## Usage tips

InstructBLIP uses the same architecture as [BLIP-2](blip2) with a tiny but important difference: it also feeds the text prompt (instruction) to the Q-Former.

> [!NOTE]
> BLIP models after release v4.46 will raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expand model embeddings layer to add special `<image>` token. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that BLIP will add the number of query tokens required per image and expand the text with as many `<image>` placeholders as there will be query tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there wil be failure when merging the embeddings.
> The attributes can be obtained from model config, as `model.config.num_query_tokens` and model embeddings expansion can be done by following [this link](https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042).

## InstructBlipConfig

### class transformers.InstructBlipConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/configuration_instructblip.py#L214)

( vision\_config = None qformer\_config = None text\_config = None num\_query\_tokens = 32 image\_token\_index = None \*\*kwargs  )

Parameters

* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [InstructBlipVisionConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipVisionConfig).
* **qformer\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [InstructBlipQFormerConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipQFormerConfig).
* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize any [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig).
* **num\_query\_tokens** (`int`, *optional*, defaults to 32) —
  The number of query tokens passed through the Transformer.
* **image\_token\_index** (`int`, *optional*) —
  Token index of special image token.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig) is the configuration class to store the configuration of a
[InstructBlipForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration). It is used to instantiate a InstructBLIP model according to the specified
arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
the defaults will yield a similar configuration to that of the InstructBLIP
[Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     InstructBlipVisionConfig,
...     InstructBlipQFormerConfig,
...     OPTConfig,
...     InstructBlipConfig,
...     InstructBlipForConditionalGeneration,
... )

>>> # Initializing a InstructBlipConfig with Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipConfig()

>>> # Initializing a InstructBlipForConditionalGeneration (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a InstructBlipConfig from a InstructBlipVisionConfig, InstructBlipQFormerConfig and any PretrainedConfig

>>> # Initializing InstructBLIP vision, InstructBLIP Q-Former and language model configurations
>>> vision_config = InstructBlipVisionConfig()
>>> qformer_config = InstructBlipQFormerConfig()
>>> text_config = OPTConfig()

>>> config = InstructBlipConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
```

#### from\_vision\_qformer\_text\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/configuration_instructblip.py#L315)

( vision\_config: InstructBlipVisionConfig qformer\_config: InstructBlipQFormerConfig text\_config: PretrainedConfig \*\*kwargs  ) → [InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)

Returns

[InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)

An instance of a configuration object

Instantiate a [InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig) (or a derived class) from a InstructBLIP vision model, Q-Former and
language model configurations.

## InstructBlipVisionConfig

### class transformers.InstructBlipVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/configuration_instructblip.py#L26)

( hidden\_size = 1408 intermediate\_size = 6144 num\_hidden\_layers = 39 num\_attention\_heads = 16 image\_size = 224 patch\_size = 14 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 initializer\_range = 1e-10 qkv\_bias = True \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1408) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 6144) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 39) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported. to 1e-5): The epsilon used by the layer
  normalization layers.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 1e-10) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries and values in the self-attention layers.

This is the configuration class to store the configuration of a [InstructBlipVisionModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipVisionModel). It is used to
instantiate a InstructBLIP vision encoder according to the specified arguments, defining the model architecture.
Instantiating a configuration defaults will yield a similar configuration to that of the InstructBLIP
[Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import InstructBlipVisionConfig, InstructBlipVisionModel

>>> # Initializing a InstructBlipVisionConfig with Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipVisionConfig()

>>> # Initializing a InstructBlipVisionModel (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## InstructBlipQFormerConfig

### class transformers.InstructBlipQFormerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/configuration_instructblip.py#L110)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' cross\_attention\_frequency = 2 encoder\_hidden\_size = 1408 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling the model.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Token id used for padding sequences.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **cross\_attention\_frequency** (`int`, *optional*, defaults to 2) —
  The frequency of adding cross-attention to the Transformer layers.
* **encoder\_hidden\_size** (`int`, *optional*, defaults to 1408) —
  The hidden size of the hidden states for cross-attention.

This is the configuration class to store the configuration of a [InstructBlipQFormerModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipQFormerModel). It is used to
instantiate a InstructBLIP Querying Transformer (Q-Former) model according to the specified arguments, defining the
model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
architecture. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs.
Read the documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Note that [InstructBlipQFormerModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipQFormerModel) is very similar to [BertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel) with interleaved cross-attention.

Examples:


```
>>> from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel

>>> # Initializing a InstructBLIP Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipQFormerConfig()

>>> # Initializing a model (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipQFormerModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## InstructBlipProcessor

### class transformers.InstructBlipProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/processing_instructblip.py#L50)

( image\_processor tokenizer qformer\_tokenizer num\_query\_tokens = None \*\*kwargs  )

Parameters

* **image\_processor** (`BlipImageProcessor`) —
  An instance of [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.
* **tokenizer** (`AutoTokenizer`) —
  An instance of [‘PreTrainedTokenizer`]. The tokenizer is a required input.
* **qformer\_tokenizer** (`AutoTokenizer`) —
  An instance of [‘PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
* **num\_query\_tokens** (`int`, *optional*) —”
  Number of tokens used by the Qformer as queries, should be same as in model’s config.

Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
processor.

[InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) and [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See the
docstring of `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## InstructBlipVisionModel

### class transformers.InstructBlipVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L451)

( config: InstructBlipVisionConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L466)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

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

The [InstructBlipVisionModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## InstructBlipQFormerModel

### class transformers.InstructBlipQFormerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L948)

( config: InstructBlipQFormerConfig  )

Querying Transformer (Q-Former), used in InstructBLIP. Slightly modified from BLIP-2 as it also takes the
instruction as input.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1026)

( input\_ids: LongTensor attention\_mask: typing.Optional[torch.FloatTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None query\_embeds: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

encoder\_hidden\_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.
encoder\_attention\_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

* 1 for tokens that are **not masked**,
* 0 for tokens that are **masked**.
  past\_key\_values (`Cache` of length `config.n_layers` with each tuple having 4 tensors of:
  shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
  value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
  used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key
  value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
  `(batch_size, sequence_length)`.
  use\_cache (`bool`, *optional*):
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

## InstructBlipModel

### class transformers.InstructBlipModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1144)

( config: InstructBlipConfig  )

Parameters

* **config** ([InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

InstructBLIP base Model consisting of language model, qformer and vision encoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1213)

( pixel\_values: FloatTensor qformer\_input\_ids: FloatTensor qformer\_attention\_mask: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **qformer\_input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the Q-Former. Input tokens can optionally be provided
  to serve as text prompt, which the Q-Former model will encode.

  Indices can be obtained using [InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor). See `InstructBlipProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **qformer\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
  provided to serve as text prompt, which the language model can continue.

  Indices can be obtained using [InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor). See `InstructBlipProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  Only relevant in case an encoder-decoder language model (like T5) is used.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Language modeling loss from the language model.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head of the language model.
* **vision\_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) — Outputs of the vision encoder.
* **qformer\_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) — Outputs of the Q-Former (Querying Transformer).
* **language\_model\_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) — Outputs of the language model.

The [InstructBlipModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1198)

( input\_ids: LongTensor inputs\_embeds: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`.

## InstructBlipForConditionalGeneration

### class transformers.InstructBlipForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1346)

( config: InstructBlipConfig  )

Parameters

* **config** ([InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision
encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1488)

( pixel\_values: FloatTensor qformer\_input\_ids: FloatTensor qformer\_attention\_mask: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **qformer\_input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the Q-Former. Input tokens can optionally be provided
  to serve as text prompt, which the Q-Former model will encode.

  Indices can be obtained using [InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor). See `InstructBlipProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **qformer\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
  provided to serve as text prompt, which the language model can continue.

  Indices can be obtained using [InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor). See `InstructBlipProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  Only relevant in case an encoder-decoder language model (like T5) is used.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Language modeling loss from the language model.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head of the language model.
* **vision\_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) — Outputs of the vision encoder.
* **qformer\_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) — Outputs of the Q-Former (Querying Transformer).
* **language\_model\_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) — Outputs of the language model.

The [InstructBlipForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
>>> import torch
>>> from PIL import Image
>>> import requests

>>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
>>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
>>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> prompt = "What is unusual about this image?"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

>>> outputs = model.generate(
...     **inputs,
...     do_sample=False,
...     num_beams=5,
...     max_length=256,
...     min_length=1,
...     top_p=0.9,
...     repetition_penalty=1.5,
...     length_penalty=1.0,
...     temperature=1,
... )
>>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblip/modeling_instructblip.py#L1639)

( pixel\_values: FloatTensor qformer\_input\_ids: typing.Optional[torch.LongTensor] = None qformer\_attention\_mask: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None interpolate\_pos\_encoding: bool = False \*\*generate\_kwargs  ) → captions (list)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape (batch\_size, num\_channels, height, width)) —
  Input images to be processed.
* **qformer\_input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  The sequence used as a prompt to be fed to the Q-Former module.
* **qformer\_attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  Mask to avoid performing attention on padding token indices.
* **input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  The sequence used as a prompt for the generation.
* **attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  Mask to avoid performing attention on padding token indices.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Embedded representation of the inputs. Should be float, not int tokens.
* **interpolate\_pos\_encoding** (`bool`, *optional*, defaults to `False`) —
  Whether to interpolate the positional encoding of the image embeddings.

Returns

captions (list)

A list of strings of length batch\_size \* num\_captions.

Overrides `generate` function to be able to use the model as a conditional generator.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/instructblip.md)
