*This model was released on 2023-01-30 and added to Hugging Face Transformers on 2023-02-09.*

# BLIP-2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The BLIP-2 model was proposed in [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://huggingface.co/papers/2301.12597) by
Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. BLIP-2 leverages frozen pre-trained image encoders and large language models (LLMs) by training a lightweight, 12-layer Transformer
encoder in between them, achieving state-of-the-art performance on various vision-language tasks. Most notably, BLIP-2 improves upon [Flamingo](https://huggingface.co/papers/2204.14198), an 80 billion parameter model, by 8.7%
on zero-shot VQAv2 with 54x fewer trainable parameters.

The abstract from the paper is the following:

*The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/blip2_architecture.jpg) BLIP-2 architecture. Taken from the [original paper.](https://huggingface.co/papers/2301.12597)

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/salesforce/LAVIS/tree/5ee63d688ba4cebff63acee04adaef2dee9af207).

## Usage tips

* BLIP-2 can be used for conditional text generation given an image and an optional text prompt. At inference time, it’s recommended to use the `generate` method.
* One can use [Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) to prepare images for the model, and decode the predicted tokens ID’s back to text.

> [!NOTE]
> BLIP models after release v4.46 will raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expand model embeddings layer to add special `<image>` token. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that BLIP will add the number of query tokens required per image and expand the text with as many `<image>` placeholders as there will be query tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there will be failure when merging the embeddings.
> The attributes can be obtained from model config, as `model.config.num_query_tokens` and model embeddings expansion can be done by following [this link](https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BLIP-2.

* Demo notebooks for BLIP-2 for image captioning, visual question answering (VQA) and chat-like conversations can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BLIP-2).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## Blip2Config

### class transformers.Blip2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/configuration_blip_2.py#L218)

( vision\_config = None qformer\_config = None text\_config = None num\_query\_tokens = 32 image\_text\_hidden\_size = 256 image\_token\_index = None \*\*kwargs  )

Parameters

* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Blip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionConfig).
* **qformer\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Blip2QFormerConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerConfig).
* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize any [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig).
* **num\_query\_tokens** (`int`, *optional*, defaults to 32) —
  The number of query tokens passed through the Transformer.
* **image\_text\_hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden state of the image-text fusion layer.
* **image\_token\_index** (`int`, *optional*) —
  Token index of special image token.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) is the configuration class to store the configuration of a [Blip2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration). It is
used to instantiate a BLIP-2 model according to the specified arguments, defining the vision model, Q-Former model
and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     Blip2VisionConfig,
...     Blip2QFormerConfig,
...     OPTConfig,
...     Blip2Config,
...     Blip2ForConditionalGeneration,
... )

>>> # Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
>>> configuration = Blip2Config()

>>> # Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
>>> model = Blip2ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a Blip2Config from a Blip2VisionConfig, Blip2QFormerConfig and any PretrainedConfig

>>> # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
>>> vision_config = Blip2VisionConfig()
>>> qformer_config = Blip2QFormerConfig()
>>> text_config = OPTConfig()

>>> config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
```

#### from\_vision\_qformer\_text\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/configuration_blip_2.py#L318)

( vision\_config: Blip2VisionConfig qformer\_config: Blip2QFormerConfig text\_config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None \*\*kwargs  ) → [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)

Parameters

* **vision\_config** (`dict`) —
  Dictionary of configuration options used to initialize [Blip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionConfig).
* **qformer\_config** (`dict`) —
  Dictionary of configuration options used to initialize [Blip2QFormerConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerConfig).
* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize any [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig).

Returns

[Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)

An instance of a configuration object

Instantiate a [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) (or a derived class) from a BLIP-2 vision model, Q-Former and language model
configurations.

## Blip2VisionConfig

### class transformers.Blip2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/configuration_blip_2.py#L28)

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
  `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported. layer\_norm\_eps (`float`, *optional*, defaults
  to 1e-5): The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries and values in the self-attention layers.

This is the configuration class to store the configuration of a [Blip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionModel). It is used to instantiate a
BLIP-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration defaults will yield a similar configuration to that of the BLIP-2
[Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Blip2VisionConfig, Blip2VisionModel

>>> # Initializing a Blip2VisionConfig with Salesforce/blip2-opt-2.7b style configuration
>>> configuration = Blip2VisionConfig()

>>> # Initializing a Blip2VisionModel (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
>>> model = Blip2VisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Blip2QFormerConfig

### class transformers.Blip2QFormerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/configuration_blip_2.py#L110)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' cross\_attention\_frequency = 2 encoder\_hidden\_size = 1408 use\_qformer\_text\_input = False \*\*kwargs  )

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
  Index to be used for padding token.
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
* **use\_qformer\_text\_input** (`bool`, *optional*, defaults to `False`) —
  Whether to use BERT-style embeddings.

This is the configuration class to store the configuration of a [Blip2QFormerModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerModel). It is used to instantiate a
BLIP-2 Querying Transformer (Q-Former) model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-2
[Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture. Configuration objects
inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from
[PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Note that [Blip2QFormerModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerModel) is very similar to [BertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel) with interleaved cross-attention.

Examples:


```
>>> from transformers import Blip2QFormerConfig, Blip2QFormerModel

>>> # Initializing a BLIP-2 Salesforce/blip2-opt-2.7b style configuration
>>> configuration = Blip2QFormerConfig()

>>> # Initializing a model (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
>>> model = Blip2QFormerModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## Blip2Processor

### class transformers.Blip2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/processing_blip_2.py#L48)

( image\_processor tokenizer num\_query\_tokens = None \*\*kwargs  )

Parameters

* **image\_processor** (`BlipImageProcessor`) —
  An instance of [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.
* **tokenizer** (`AutoTokenizer`) —
  An instance of [‘PreTrainedTokenizer`]. The tokenizer is a required input.
* **num\_query\_tokens** (`int`, *optional*) —
  Number of tokens used by the Qformer as queries, should be same as in model’s config.

Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

[BlipProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) and [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See the docstring
of `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Blip2VisionModel

### class transformers.Blip2VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L535)

( config: Blip2VisionConfig  )

Parameters

* **config** ([Blip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Blip 2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L550)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
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
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

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

The [Blip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Blip2QFormerModel

### class transformers.Blip2QFormerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1036)

( config: Blip2QFormerConfig  )

Parameters

* **config** ([Blip2QFormerConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP-2 Querying Transformer (Q-Former).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1110)

( query\_embeds: FloatTensor query\_length: typing.Optional[int] = None attention\_mask: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **query\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Hidden states to be used in the attention computation. If cross-attention,
  will be used for the query (i.e., key and value will use the encoder\_hidden\_states).
* **query\_length** (`int`, *optional*) —
  Length of the query, usually based on the number of query tokens.
  If no value is provided, query\_length will be inferred by the query\_embeds.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

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
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

The [Blip2QFormerModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Blip2Model

### class transformers.Blip2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1220)

( config: Blip2Config  )

Parameters

* **config** ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer
(Q-Former) and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1473)

( pixel\_values: FloatTensor input\_ids: FloatTensor attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
  provided to serve as text prompt, which the language model can continue.

  Indices can be obtained using [Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor). See `Blip2Processor.__call__()` for details.

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
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

`transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Language modeling loss from the language model.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head of the language model.
* **vision\_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) — Outputs of the vision encoder.
* **qformer\_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) — Outputs of the Q-Former (Querying Transformer).
* **language\_model\_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) — Outputs of the language model.

The [Blip2Model](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2Model
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", dtype=torch.float16)
>>> model.to(device)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "Question: how many cats are there? Answer:"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> outputs = model(**inputs)
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1272)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → text\_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`)

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

  To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
  Training](./t5#training).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`)

The language model outputs. If `return_dict=True`, the output is a `CausalLMOutputWithPast` that
contains the language model logits, the past key values and the hidden states if
`output_hidden_states=True`.

Examples:


```
>>> import torch
>>> from transformers import AutoTokenizer, Blip2Model

>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1351)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → vision\_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
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

vision\_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`)

The vision model outputs. If `return_dict=True`, the output is a `BaseModelOutputWithPooling` that
contains the image features, the pooled image features and the hidden states if
`output_hidden_states=True`.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Blip2Model

>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_outputs = model.get_image_features(**inputs)
```

#### get\_qformer\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1397)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → vision\_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
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

vision\_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`)

The vision model outputs. If `return_dict=True`, the output is a `BaseModelOutputWithPooling` that
contains the image features, the pooled image features and the hidden states if
`output_hidden_states=True`.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2Model

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")
>>> qformer_outputs = model.get_qformer_features(**inputs)
```

## Blip2ForConditionalGeneration

### class transformers.Blip2ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1830)

( config: Blip2Config  )

Parameters

* **config** ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision
encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1963)

( pixel\_values: FloatTensor input\_ids: LongTensor attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
  provided to serve as text prompt, which the language model can continue.

  Indices can be obtained using [Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor). See `Blip2Processor.__call__()` for details.

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Language modeling loss from the language model.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head of the language model.
* **vision\_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) — Outputs of the vision encoder.
* **qformer\_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) — Outputs of the Q-Former (Querying Transformer).
* **language\_model\_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) — Outputs of the language model.

The [Blip2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

Prepare processor, model and image input


```
>>> from PIL import Image
>>> import requests
>>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained(
...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, dtype=torch.float16
... )  # doctest: +IGNORE_RESULT

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
```

Image captioning (without providing a text prompt):


```
>>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
two cats laying on a couch
```

Visual question answering (prompt = question):


```
>>> prompt = "Question: how many cats are there? Answer:"
>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

>>> generated_ids = model.generate(**inputs)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
two
```

Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
This greatly reduces the amount of memory used by the model while maintaining the same performance.


```
>>> model = Blip2ForConditionalGeneration.from_pretrained(
...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, dtype=torch.bfloat16
... )  # doctest: +IGNORE_RESULT

>>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

>>> generated_ids = model.generate(**inputs)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
two
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L2126)

( pixel\_values: FloatTensor input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None interpolate\_pos\_encoding: bool = False \*\*generate\_kwargs  ) → captions (list)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape (batch\_size, num\_channels, height, width)) —
  Input images to be processed.
* **input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  The sequence used as a prompt for the generation.
* **attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  Mask to avoid performing attention on padding token indices
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Embedded representation of the inputs. Should be float, not int tokens.
* **interpolate\_pos\_encoding** (`bool`, *optional*, defaults to `False`) —
  Whether to interpolate the positional encoding of the image embeddings.

Returns

captions (list)

A list of strings of length batch\_size \* num\_captions.

Overrides `generate` function to be able to use the model as a conditional generator.

## Blip2ForImageTextRetrieval

### class transformers.Blip2ForImageTextRetrieval

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L2220)

( config: Blip2Config  )

Parameters

* **config** ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BLIP-2 Model with a vision and text projector, and a classification head on top. The model is used in the context
of image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
the image.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L2253)

( pixel\_values: FloatTensor input\_ids: LongTensor attention\_mask: typing.Optional[torch.LongTensor] = None use\_image\_text\_matching\_head: typing.Optional[bool] = False output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.blip_2.modeling_blip_2.Blip2ImageTextMatchingModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
  provided to serve as text prompt, which the language model can continue.

  Indices can be obtained using [Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor). See `Blip2Processor.__call__()` for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **use\_image\_text\_matching\_head** (`bool`, *optional*) —
  Whether to return the Image-Text Matching or Contrastive scores.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.blip_2.modeling_blip_2.Blip2ImageTextMatchingModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.blip_2.modeling_blip_2.Blip2ImageTextMatchingModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output.
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [Blip2QFormerModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [Blip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionModel).

The [Blip2ForImageTextRetrieval](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForImageTextRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Blip2ForImageTextRetrieval

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", dtype=torch.float16)
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

>>> model.to(device)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "two cats laying on a pink blanket"

>>> inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
>>> itm_out = model(**inputs, use_image_text_matching_head=True)
>>> logits_per_image = torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

>>> print(f"{probs[0][0]:.1%} that image 0 is not '{text}'")
26.9% that image 0 is not 'two cats laying on a pink blanket'

>>> print(f"{probs[0][1]:.1%} that image 0 is '{text}'")
73.0% that image 0 is 'two cats laying on a pink blanket'

>>> texts = ["a photo of a cat", "a photo of a dog"]

>>> inputs = processor(images=image, text=texts, return_tensors="pt").to(device, torch.float16)
>>> itc_out = model(**inputs, use_image_text_matching_head=False)
>>> logits_per_image = itc_out.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

>>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
55.3% that image 0 is 'a photo of a cat'

>>> print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
44.7% that image 0 is 'a photo of a dog'
```

## Blip2TextModelWithProjection

### class transformers.Blip2TextModelWithProjection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1622)

( config: Blip2Config  )

Parameters

* **config** ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Blip 2 Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1646)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.blip_2.modeling_blip_2.Blip2TextModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.blip_2.modeling_blip_2.Blip2TextModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.blip_2.modeling_blip_2.Blip2TextModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The text embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Blip2TextModelWithProjection](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2TextModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import AutoProcessor, Blip2TextModelWithProjection

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Blip2TextModelWithProjection.from_pretrained(
...     "Salesforce/blip2-itm-vit-g", dtype=torch.float16
... )

>>> model.to(device)
>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt").to(device)

>>> outputs = model(**inputs)
>>> text_embeds = outputs.text_embeds
>>> print(text_embeds.shape)
torch.Size([2, 7, 256])
```

## Blip2VisionModelWithProjection

### class transformers.Blip2VisionModelWithProjection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1715)

( config: Blip2Config  )

Parameters

* **config** ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Blip 2 Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blip_2/modeling_blip_2.py#L1737)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.blip_2.modeling_blip_2.Blip2VisionModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) uses
  [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.blip_2.modeling_blip_2.Blip2VisionModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.blip_2.modeling_blip_2.Blip2VisionModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config)) and inputs.

* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The image embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Blip2VisionModelWithProjection](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2VisionModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Blip2VisionModelWithProjection

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
>>> model = Blip2VisionModelWithProjection.from_pretrained(
...     "Salesforce/blip2-itm-vit-g", dtype=torch.float16
... )
>>> model.to(device)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

>>> outputs = model(**inputs)
>>> image_embeds = outputs.image_embeds
>>> print(image_embeds.shape)
torch.Size([1, 32, 256])
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/blip-2.md)
