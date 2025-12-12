# InstructBLIP

## Overview

The InstructBLIP model was proposed in [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://huggingface.co/papers/2305.06500) by Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.
InstructBLIP leverages the [BLIP-2](blip2) architecture for visual instruction tuning.

The abstract from the paper is the following:

*General-purpose language models that can solve various language-domain tasks have emerged driven by the pre-training and instruction-tuning pipeline. However, building general-purpose vision-language models is challenging due to the increased task discrepancy introduced by the additional visual input. Although vision-language pre-training has been widely studied, vision-language instruction tuning remains relatively less explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pre-trained BLIP-2 models. We gather a wide variety of 26 publicly available datasets, transform them into instruction tuning format and categorize them into two clusters for held-in instruction tuning and held-out zero-shot evaluation. Additionally, we introduce instruction-aware visual feature extraction, a crucial method that enables the model to extract informative features tailored to the given instruction. The resulting InstructBLIP models achieve state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and the larger Flamingo. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA IMG). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models.*

 InstructBLIP architecture. Taken from the original paper. 

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

## Usage tips

InstructBLIP uses the same architecture as [BLIP-2](blip2) with a tiny but important difference: it also feeds the text prompt (instruction) to the Q-Former.

> [!NOTE]
> BLIP models after release v4.46 will raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expand model embeddings layer to add special `` token. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that BLIP will add the number of query tokens required per image and expand the text with as many `` placeholders as there will be query tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there wil be failure when merging the embeddings.
The attributes can be obtained from model config, as `model.config.num_query_tokens` and model embeddings expansion can be done by following [this link](https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042).

## InstructBlipConfig[[transformers.InstructBlipConfig]]

#### transformers.InstructBlipConfig[[transformers.InstructBlipConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/configuration_instructblip.py#L206)

[InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig) is the configuration class to store the configuration of a
[InstructBlipForConditionalGeneration](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration). It is used to instantiate a InstructBLIP model according to the specified
arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
the defaults will yield a similar configuration to that of the InstructBLIP
[Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
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

>>> # We can also initialize a InstructBlipConfig from a InstructBlipVisionConfig, InstructBlipQFormerConfig and any PreTrainedConfig

>>> # Initializing InstructBLIP vision, InstructBLIP Q-Former and language model configurations
>>> vision_config = InstructBlipVisionConfig()
>>> qformer_config = InstructBlipQFormerConfig()
>>> text_config = OPTConfig()

>>> config = InstructBlipConfig(vision_config=vision_config, qformer_config=qformer_config, text_config=text_config)
```

**Parameters:**

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [InstructBlipVisionConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipVisionConfig).

qformer_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [InstructBlipQFormerConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipQFormerConfig).

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize any [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig).

num_query_tokens (`int`, *optional*, defaults to 32) : The number of query tokens passed through the Transformer. 

image_token_index (`int`, *optional*) : Token index of special image token.

kwargs (*optional*) : Dictionary of keyword arguments.

## InstructBlipVisionConfig[[transformers.InstructBlipVisionConfig]]

#### transformers.InstructBlipVisionConfig[[transformers.InstructBlipVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/configuration_instructblip.py#L26)

This is the configuration class to store the configuration of a [InstructBlipVisionModel](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipVisionModel). It is used to
instantiate a InstructBLIP vision encoder according to the specified arguments, defining the model architecture.
Instantiating a configuration defaults will yield a similar configuration to that of the InstructBLIP
[Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import InstructBlipVisionConfig, InstructBlipVisionModel

>>> # Initializing a InstructBlipVisionConfig with Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipVisionConfig()

>>> # Initializing a InstructBlipVisionModel (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 1408) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 6144) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 39) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 14) : The size (resolution) of each patch.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported. to 1e-5): The epsilon used by the layer normalization layers.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 1e-10) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries and values in the self-attention layers.

## InstructBlipQFormerConfig[[transformers.InstructBlipQFormerConfig]]

#### transformers.InstructBlipQFormerConfig[[transformers.InstructBlipQFormerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/configuration_instructblip.py#L110)

This is the configuration class to store the configuration of a [InstructBlipQFormerModel](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipQFormerModel). It is used to
instantiate a InstructBLIP Querying Transformer (Q-Former) model according to the specified arguments, defining the
model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
architecture. Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs.
Read the documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Note that [InstructBlipQFormerModel](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipQFormerModel) is very similar to [BertLMHeadModel](/docs/transformers/main/en/model_doc/bert#transformers.BertLMHeadModel) with interleaved cross-attention.

Examples:

```python
>>> from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel

>>> # Initializing a InstructBLIP Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipQFormerConfig()

>>> # Initializing a model (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipQFormerModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30522) : Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling the model.

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

pad_token_id (`int`, *optional*, defaults to 0) : Token id used for padding sequences.

cross_attention_frequency (`int`, *optional*, defaults to 2) : The frequency of adding cross-attention to the Transformer layers.

encoder_hidden_size (`int`, *optional*, defaults to 1408) : The hidden size of the hidden states for cross-attention.

## InstructBlipProcessor[[transformers.InstructBlipProcessor]]

#### transformers.InstructBlipProcessor[[transformers.InstructBlipProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/processing_instructblip.py#L47)

Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
processor.

[InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor) offers all the functionalities of [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) and [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See the
docstring of `__call__()` and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor (`BlipImageProcessor`) : An instance of [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). The image processor is a required input.

tokenizer (`AutoTokenizer`) : An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.

qformer_tokenizer (`AutoTokenizer`) : An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.

num_query_tokens (`int`, *optional*) --" Number of tokens used by the Qformer as queries, should be same as in model's config.

## InstructBlipVisionModel[[transformers.InstructBlipVisionModel]]

#### transformers.InstructBlipVisionModel[[transformers.InstructBlipVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L374)

forwardtransformers.InstructBlipVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L394[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [InstructBlipVisionModel](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## InstructBlipQFormerModel[[transformers.InstructBlipQFormerModel]]

#### transformers.InstructBlipQFormerModel[[transformers.InstructBlipQFormerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L768)

Querying Transformer (Q-Former), used in InstructBLIP. Slightly modified from BLIP-2 as it also takes the
instruction as input.

forwardtransformers.InstructBlipQFormerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L848[{"name": "input_ids", "val": ": LongTensor"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "query_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **query_embeds** (`torch.FloatTensor`  of shape `(batch_size, sequence_length, hidden_size)`) --
  Hidden states to be used in the attention computation. If cross-attention,
  will be used for the query (i.e., key and value will use the encoder_hidden_states).
- **encoder_hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
- **encoder_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.0[transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
The [InstructBlipQFormerModel](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipQFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

query_embeds (`torch.FloatTensor`  of shape `(batch_size, sequence_length, hidden_size)`) : Hidden states to be used in the attention computation. If cross-attention, will be used for the query (i.e., key and value will use the encoder_hidden_states).

encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.

encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

## InstructBlipModel[[transformers.InstructBlipModel]]

#### transformers.InstructBlipModel[[transformers.InstructBlipModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L928)

InstructBLIP base Model consisting of language model, qformer and vision encoder.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InstructBlipModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L992[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "qformer_input_ids", "val": ": FloatTensor"}, {"name": "qformer_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "input_ids", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **qformer_input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary of the Q-Former. Input tokens can optionally be provided
  to serve as text prompt, which the Q-Former model will encode.

  Indices can be obtained using [InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor). See `InstructBlipProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
- **qformer_attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **input_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  Only relevant in case an encoder-decoder language model (like T5) is used.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0`transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Language modeling loss from the language model.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head of the language model.
- **vision_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) -- Outputs of the vision encoder.
- **qformer_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) -- Outputs of the Q-Former (Querying Transformer).
- **language_model_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) -- Outputs of the language model.
The [InstructBlipModel](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Language modeling loss from the language model.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head of the language model.
- **vision_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) -- Outputs of the vision encoder.
- **qformer_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) -- Outputs of the Q-Former (Querying Transformer).
- **language_model_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) -- Outputs of the language model.
#### get_placeholder_mask[[transformers.InstructBlipModel.get_placeholder_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L977)

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`.

## InstructBlipForConditionalGeneration[[transformers.InstructBlipForConditionalGeneration]]

#### transformers.InstructBlipForConditionalGeneration[[transformers.InstructBlipForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L1100)

InstructBLIP Model for generating text given an image and an optional text prompt. The model consists of a vision
encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InstructBlipForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L1239[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "qformer_input_ids", "val": ": FloatTensor"}, {"name": "qformer_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "input_ids", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor). See [BlipImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor) uses
  [BlipImageProcessor](/docs/transformers/main/en/model_doc/blip#transformers.BlipImageProcessor) for processing images).
- **qformer_input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary of the Q-Former. Input tokens can optionally be provided
  to serve as text prompt, which the Q-Former model will encode.

  Indices can be obtained using [InstructBlipProcessor](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipProcessor). See `InstructBlipProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
- **qformer_attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **input_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  Only relevant in case an encoder-decoder language model (like T5) is used.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
  1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
  config.vocab_size]`
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.0`transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Language modeling loss from the language model.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head of the language model.
- **vision_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) -- Outputs of the vision encoder.
- **qformer_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) -- Outputs of the Q-Former (Querying Transformer).
- **language_model_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) -- Outputs of the language model.
The [InstructBlipForConditionalGeneration](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
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

**Parameters:**

config ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipConfig](/docs/transformers/main/en/model_doc/instructblip#transformers.InstructBlipConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Language modeling loss from the language model.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head of the language model.
- **vision_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) -- Outputs of the vision encoder.
- **qformer_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) -- Outputs of the Q-Former (Querying Transformer).
- **language_model_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) -- Outputs of the language model.
#### generate[[transformers.InstructBlipForConditionalGeneration.generate]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/instructblip/modeling_instructblip.py#L1368)

Overrides `generate` function to be able to use the model as a conditional generator.

**Parameters:**

pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)) : Input images to be processed.

qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*) : The sequence used as a prompt to be fed to the Q-Former module.

qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*) : Mask to avoid performing attention on padding token indices.

input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*) : The sequence used as a prompt for the generation.

attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*) : Mask to avoid performing attention on padding token indices.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Embedded representation of the inputs. Should be float, not int tokens.

interpolate_pos_encoding (`bool`, *optional*, defaults to `False`) : Whether to interpolate the positional encoding of the image embeddings.

**Returns:**

`captions (list)`

A list of strings of length batch_size * num_captions.
