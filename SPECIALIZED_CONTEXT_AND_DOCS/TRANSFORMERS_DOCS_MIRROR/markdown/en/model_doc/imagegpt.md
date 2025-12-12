*This model was released on 2020-06-17 and added to Hugging Face Transformers on 2021-11-18.*

# ImageGPT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The ImageGPT model was proposed in [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt) by Mark
Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever. ImageGPT (iGPT) is a GPT-2-like
model trained to predict the next pixel value, allowing for both unconditional and conditional image generation.

The abstract from the [paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V1_ICML.pdf) is the following:

*Inspired by progress in unsupervised representation learning for natural language, we examine whether similar models
can learn useful representations for images. We train a sequence Transformer to auto-regressively predict pixels,
without incorporating knowledge of the 2D input structure. Despite training on low-resolution ImageNet without labels,
we find that a GPT-2 scale model learns strong image representations as measured by linear probing, fine-tuning, and
low-data classification. On CIFAR-10, we achieve 96.3% accuracy with a linear probe, outperforming a supervised Wide
ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models. We are also
competitive with self-supervised benchmarks on ImageNet when substituting pixels for a VQVAE encoding, achieving 69.0%
top-1 accuracy on a linear probe of our features.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png) Summary of the approach. Taken from the [original paper](https://cdn.openai.com/papers/Generative\_Pretraining\_from\_Pixels\_V2.pdf).

This model was contributed by [nielsr](https://huggingface.co/nielsr), based on [this issue](https://github.com/openai/image-gpt/issues/7). The original code can be found
[here](https://github.com/openai/image-gpt).

## Usage tips

* ImageGPT is almost exactly the same as [GPT-2](gpt2), with the exception that a different activation
  function is used (namely “quick gelu”), and the layer normalization layers don’t mean center the inputs. ImageGPT
  also doesn’t have tied input- and output embeddings.
* As the time- and memory requirements of the attention mechanism of Transformers scales quadratically in the sequence
  length, the authors pre-trained ImageGPT on smaller input resolutions, such as 32x32 and 64x64. However, feeding a
  sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large. Therefore, the authors
  applied k-means clustering to the (R,G,B) pixel values with k=512. This way, we only have a 32\*32 = 1024-long
  sequence, but now of integers in the range 0..511. So we are shrinking the sequence length at the cost of a bigger
  embedding matrix. In other words, the vocabulary size of ImageGPT is 512, + 1 for a special “start of sentence” (SOS)
  token, used at the beginning of every sequence. One can use [ImageGPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTImageProcessor) to prepare
  images for the model.
* Despite being pre-trained entirely unsupervised (i.e. without the use of any labels), ImageGPT produces fairly
  performant image features useful for downstream tasks, such as image classification. The authors showed that the
  features in the middle of the network are the most performant, and can be used as-is to train a linear model (such as
  a sklearn logistic regression model for example). This is also referred to as “linear probing”. Features can be
  easily obtained by first forwarding the image through the model, then specifying `output_hidden_states=True`, and
  then average-pool the hidden states at whatever layer you like.
* Alternatively, one can further fine-tune the entire model on a downstream dataset, similar to BERT. For this, you can
  use [ImageGPTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification).
* ImageGPT comes in different sizes: there’s ImageGPT-small, ImageGPT-medium and ImageGPT-large. The authors did also
  train an XL variant, which they didn’t release. The differences in size are summarized in the following table:

| **Model variant** | **Depths** | **Hidden sizes** | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
| --- | --- | --- | --- | --- | --- |
| MiT-b0 | [2, 2, 2, 2] | [32, 64, 160, 256] | 256 | 3.7 | 70.5 |
| MiT-b1 | [2, 2, 2, 2] | [64, 128, 320, 512] | 256 | 14.0 | 78.7 |
| MiT-b2 | [3, 4, 6, 3] | [64, 128, 320, 512] | 768 | 25.4 | 81.6 |
| MiT-b3 | [3, 4, 18, 3] | [64, 128, 320, 512] | 768 | 45.2 | 83.1 |
| MiT-b4 | [3, 8, 27, 3] | [64, 128, 320, 512] | 768 | 62.6 | 83.6 |
| MiT-b5 | [3, 6, 40, 3] | [64, 128, 320, 512] | 768 | 82.0 | 83.8 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ImageGPT.

Image Classification

* Demo notebooks for ImageGPT can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ImageGPT).
* [ImageGPTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
* See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ImageGPTConfig

### class transformers.ImageGPTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/configuration_imagegpt.py#L32)

( vocab\_size = 513 n\_positions = 1024 n\_embd = 512 n\_layer = 24 n\_head = 8 n\_inner = None activation\_function = 'quick\_gelu' resid\_pdrop = 0.1 embd\_pdrop = 0.1 attn\_pdrop = 0.1 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 scale\_attn\_weights = True use\_cache = True tie\_word\_embeddings = False scale\_attn\_by\_inverse\_layer\_idx = False reorder\_and\_upcast\_attn = False \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 512) —
  Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [ImageGPTModel](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel) or `TFImageGPTModel`.
* **n\_positions** (`int`, *optional*, defaults to 32\*32) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **n\_embd** (`int`, *optional*, defaults to 512) —
  Dimensionality of the embeddings and hidden states.
* **n\_layer** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **n\_head** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **n\_inner** (`int`, *optional*, defaults to None) —
  Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n\_embd
* **activation\_function** (`str`, *optional*, defaults to `"quick_gelu"`) —
  Activation function (can be one of the activation functions defined in src/transformers/activations.py).
  Defaults to “quick\_gelu”.
* **resid\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **embd\_pdrop** (`int`, *optional*, defaults to 0.1) —
  The dropout ratio for the embeddings.
* **attn\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-5) —
  The epsilon to use in the layer normalization layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **scale\_attn\_weights** (`bool`, *optional*, defaults to `True`) —
  Scale attention weights by dividing by sqrt(hidden\_size)..
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **scale\_attn\_by\_inverse\_layer\_idx** (`bool`, *optional*, defaults to `False`) —
  Whether to additionally scale attention weights by `1 / layer_idx + 1`.
* **reorder\_and\_upcast\_attn** (`bool`, *optional*, defaults to `False`) —
  Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
  dot-product/softmax to float() when training with mixed precision.

This is the configuration class to store the configuration of a [ImageGPTModel](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel) or a `TFImageGPTModel`. It is
used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
[openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ImageGPTConfig, ImageGPTModel

>>> # Initializing a ImageGPT configuration
>>> configuration = ImageGPTConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = ImageGPTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ImageGPTFeatureExtractor

### class transformers.ImageGPTFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/feature_extraction_imagegpt.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## ImageGPTImageProcessor

### class transformers.ImageGPTImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/image_processing_imagegpt.py#L61)

( clusters: typing.Union[list[list[int]], numpy.ndarray, NoneType] = None do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_normalize: bool = True do\_color\_quantize: bool = True \*\*kwargs  )

Parameters

* **clusters** (`np.ndarray` or `list[list[int]]`, *optional*) —
  The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overridden by `clusters`
  in `preprocess`.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s dimensions to `(size["height"], size["width"])`. Can be overridden by
  `do_resize` in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 256, "width": 256}`):
  Size of the image after resizing. Can be overridden by `size` in `preprocess`.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image pixel value to between [-1, 1]. Can be overridden by `do_normalize` in
  `preprocess`.
* **do\_color\_quantize** (`bool`, *optional*, defaults to `True`) —
  Whether to color quantize the image. Can be overridden by `do_color_quantize` in `preprocess`.

Constructs a ImageGPT image processor. This image processor can be used to resize images to a smaller resolution
(such as 32x32 or 64x64), normalize them and finally color quantize them to obtain sequences of “pixel values”
(color clusters).

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/image_processing_imagegpt.py#L178)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_normalize: typing.Optional[bool] = None do\_color\_quantize: typing.Optional[bool] = None clusters: typing.Union[list[list[int]], numpy.ndarray, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_normalize=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image
* **do\_color\_quantize** (`bool`, *optional*, defaults to `self.do_color_quantize`) —
  Whether to color quantize the image.
* **clusters** (`np.ndarray` or `list[list[int]]`, *optional*, defaults to `self.clusters`) —
  Clusters used to quantize the image of shape `(n_clusters, 3)`. Only has an effect if
  `do_color_quantize` is set to `True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    Only has an effect if `do_color_quantize` is set to `False`.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## ImageGPTModel

### class transformers.ImageGPTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L539)

( config: ImageGPTConfig  )

Parameters

* **config** ([ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Imagegpt Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L572)

( input\_ids: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs: typing.Any  ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) —
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
  `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
  sequence tokens in the vocabulary.

  If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [ImageGPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_hidden\_states** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
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
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [ImageGPTModel](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ImageGPTModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
>>> model = ImageGPTModel.from_pretrained("openai/imagegpt-small")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ImageGPTForCausalImageModeling

### class transformers.ImageGPTForCausalImageModeling

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L785)

( config: ImageGPTConfig  )

Parameters

* **config** ([ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L799)

( input\_ids: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs: typing.Any  ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) —
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
  `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
  sequence tokens in the vocabulary.

  If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [ImageGPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_hidden\_states** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
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
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig)) and inputs.

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

The [ImageGPTForCausalImageModeling](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForCausalImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
>>> import torch
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
>>> model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> model.to(device)
>>> # unconditional generation of 8 images
>>> batch_size = 4
>>> context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token
>>> context = context.to(device)
>>> output = model.generate(
...     input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
... )

>>> clusters = image_processor.clusters
>>> height = image_processor.size["height"]
>>> width = image_processor.size["width"]

>>> samples = output[:, 1:].detach().cpu().numpy()
>>> samples_img = [
...     np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples
... ]  # convert color cluster tokens back to pixels
>>> f, axes = plt.subplots(1, batch_size, dpi=300)

>>> for img, ax in zip(samples_img, axes):
...     ax.axis("off")
...     ax.imshow(img)
```

## ImageGPTForImageClassification

### class transformers.ImageGPTForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L921)

( config: ImageGPTConfig  )

Parameters

* **config** ([ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ImageGPT Model transformer with an image classification head on top (linear layer).
[ImageGPTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) average-pools the hidden states in order to do the classification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L931)

( input\_ids: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs: typing.Any  ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) —
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
  `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
  sequence tokens in the vocabulary.

  If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [ImageGPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

`transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ImageGPTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ImageGPTForImageClassification
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
>>> model = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/imagegpt.md)
