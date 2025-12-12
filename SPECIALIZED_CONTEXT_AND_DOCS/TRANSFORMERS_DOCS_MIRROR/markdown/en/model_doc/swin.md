*This model was released on 2021-03-25 and added to Hugging Face Transformers on 2022-01-21.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Swin Transformer

[Swin Transformer](https://huggingface.co/papers/2103.14030) is a hierarchical vision transformer. Images are processed in patches and windowed self-attention is used to capture local information. These windows are shifted across the image to allow for cross-window connections, capturing global information more efficiently. This hierarchical approach with shifted windows allows the Swin Transformer to process images effectively at different scales and achieve linear computational complexity relative to image size, making it a versatile backbone for various vision tasks like image classification and object detection.

You can find all official Swin Transformer checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=swin) organization.

Click on the Swin Transformer models in the right sidebar for more examples of how to apply Swin Transformer to different image tasks.

The example below demonstrates how to classify an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/swin-tiny-patch4-window7-224",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

## Notes

* Swin can pad the inputs for any input height and width divisible by `32`.
* Swin can be used as a [backbone](../backbones). When `output_hidden_states = True`, it outputs both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, sequence_length, num_channels)`.

## SwinConfig

### class transformers.SwinConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/configuration_swin.py#L31)

( image\_size = 224 patch\_size = 4 num\_channels = 3 embed\_dim = 96 depths = [2, 2, 6, 2] num\_heads = [3, 6, 12, 24] window\_size = 7 mlp\_ratio = 4.0 qkv\_bias = True hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 drop\_path\_rate = 0.1 hidden\_act = 'gelu' use\_absolute\_embeddings = False initializer\_range = 0.02 layer\_norm\_eps = 1e-05 encoder\_stride = 32 out\_features = None out\_indices = None \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 4) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **embed\_dim** (`int`, *optional*, defaults to 96) —
  Dimensionality of patch embedding.
* **depths** (`list(int)`, *optional*, defaults to `[2, 2, 6, 2]`) —
  Depth of each layer in the Transformer encoder.
* **num\_heads** (`list(int)`, *optional*, defaults to `[3, 6, 12, 24]`) —
  Number of attention heads in each layer of the Transformer encoder.
* **window\_size** (`int`, *optional*, defaults to 7) —
  Size of windows.
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  Ratio of MLP hidden dimensionality to embedding dimensionality.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not a learnable bias should be added to the queries, keys and values.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings and encoder.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.1) —
  Stochastic depth rate.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
  `"selu"` and `"gelu_new"` are supported.
* **use\_absolute\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add absolute position embeddings to the patch embeddings.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **encoder\_stride** (`int`, *optional*, defaults to 32) —
  Factor to increase the spatial resolution by in the decoder head for masked image modeling.
* **out\_features** (`list[str]`, *optional*) —
  If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
  (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
  corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
  same order as defined in the `stage_names` attribute.
* **out\_indices** (`list[int]`, *optional*) —
  If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
  many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
  If unset and `out_features` is unset, will default to the last stage. Must be in the
  same order as defined in the `stage_names` attribute.

This is the configuration class to store the configuration of a [SwinModel](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinModel). It is used to instantiate a Swin
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Swin
[microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SwinConfig, SwinModel

>>> # Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
>>> configuration = SwinConfig()

>>> # Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
>>> model = SwinModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SwinModel

### class transformers.SwinModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/modeling_swin.py#L878)

( config add\_pooling\_layer = True use\_mask\_token = False  )

Parameters

* **config** ([SwinModel](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether or not to apply pooling layer.
* **use\_mask\_token** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create and apply mask tokens in the embedding layer.

The bare Swin Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/modeling_swin.py#L911)

( pixel\_values: typing.Optional[torch.FloatTensor] = None bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.swin.modeling_swin.SwinModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
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

`transformers.models.swin.modeling_swin.SwinModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.swin.modeling_swin.SwinModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed) — Average pooling of the last layer hidden-state.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **reshaped\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.

The [SwinModel](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## SwinForMaskedImageModeling

### class transformers.SwinForMaskedImageModeling

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/modeling_swin.py#L989)

( config  )

Parameters

* **config** ([SwinForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForMaskedImageModeling)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://huggingface.co/papers/2111.09886).

Note that we provide a script to pre-train this model on custom data in our [examples
directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/modeling_swin.py#L1006)

( pixel\_values: typing.Optional[torch.FloatTensor] = None bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.swin.modeling_swin.SwinMaskedImageModelingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
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

`transformers.models.swin.modeling_swin.SwinMaskedImageModelingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.swin.modeling_swin.SwinMaskedImageModelingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided) — Masked image modeling (MLM) loss.
* **reconstruction** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Reconstructed pixel values.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **reshaped\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.

The [SwinForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, SwinForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
>>> model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> # create random boolean mask of shape (batch_size, num_patches)
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 192, 192]
```

## SwinForImageClassification

### class transformers.SwinForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/modeling_swin.py#L1106)

( config  )

Parameters

* **config** ([SwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

Note that it’s possible to fine-tune Swin on higher resolution images than the ones it has been trained on, by
setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
position embeddings to the higher resolution.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/swin/modeling_swin.py#L1121)

( pixel\_values: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.swin.modeling_swin.SwinImageClassifierOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

`transformers.models.swin.modeling_swin.SwinImageClassifierOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.swin.modeling_swin.SwinImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **reshaped\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.

The [SwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, SwinForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/swin.md)
