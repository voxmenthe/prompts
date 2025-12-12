*This model was released on 2025-08-13 and added to Hugging Face Transformers on 2025-08-14.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# DINOv3

[DINOv3](https://huggingface.co/papers/2508.10104) is a family of versatile vision foundation models that outperforms the specialized state of the art across a broad range of settings, without fine-tuning. DINOv3 produces high-quality dense features that achieve outstanding performance on various vision tasks, significantly surpassing previous self- and weakly-supervised foundation models.

You can find all the original DINOv3 checkpoints under the [DINOv3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) collection.

Click on the DINOv3 models in the right sidebar for more examples of how to apply DINOv3 to different vision tasks.

The example below demonstrates how to obtain an image embedding with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-feature-extraction",
    model="facebook/dinov3-vits16-pretrain-lvd1689m",
    dtype=torch.bfloat16,
)

pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from torchao.quantization import Int4WeightOnlyConfig
from transformers.image_utils import load_image


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitsplus-pretrain-lvd1689m")

quant_type = Int4WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_type)

model = AutoModel.from_pretrained(
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
```

## Notes

* The example below shows how to split the output tensor into:

  + one embedding for the whole image, commonly referred to as a `CLS` token,
    useful for classification and retrieval
  + register tokens - learnable embeddings that act as dedicated “memory slots” for global information,
    they reduce high-norm artifacts in patch tokens, yielding cleaner attention maps and better
    performance on dense prediction tasks.
  + a set of local embeddings, one for each `16x16` patch of the input image,
    useful for dense tasks, such as semantic segmentation


  ```
  import torch
  from transformers import AutoImageProcessor, AutoModel
  from transformers.image_utils import load_image

  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = load_image(url)
  print("Image size:", image.height, image.width)  # [480, 640]

  processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
  model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
  patch_size = model.config.patch_size
  print("Patch size:", patch_size) # 16
  print("Num register tokens:", model.config.num_register_tokens) # 4

  inputs = processor(images=image, return_tensors="pt")
  print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]

  batch_size, _, img_height, img_width = inputs.pixel_values.shape
  num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
  num_patches_flat = num_patches_height * num_patches_width

  with torch.inference_mode():
    outputs = model(**inputs)

  last_hidden_states = outputs.last_hidden_state
  print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
  assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

  cls_token = last_hidden_states[:, 0, :]
  patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
  patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
  ```

## DINOv3ViTConfig

### class transformers.DINOv3ViTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_vit/configuration_dinov3_vit.py#L26)

( patch\_size: int = 16 hidden\_size: int = 384 intermediate\_size: int = 1536 num\_hidden\_layers: int = 12 num\_attention\_heads: int = 6 hidden\_act: str = 'gelu' attention\_dropout: float = 0.0 initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-05 rope\_theta: float = 100.0 image\_size: int = 224 num\_channels: int = 3 query\_bias: bool = True key\_bias: bool = False value\_bias: bool = True proj\_bias: bool = True mlp\_bias: bool = True layerscale\_value: float = 1.0 drop\_path\_rate: float = 0.0 use\_gated\_mlp: bool = False num\_register\_tokens: int = 0 pos\_embed\_shift: typing.Optional[float] = None pos\_embed\_jitter: typing.Optional[float] = None pos\_embed\_rescale: typing.Optional[float] = 2.0 \*\*kwargs  )

Parameters

* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **hidden\_size** (`int`, *optional*, defaults to 384) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 6) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **rope\_theta** (`float`, *optional*, defaults to 100.0) —
  The base period of the RoPE embeddings.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **query\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the query projection.
* **key\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the key projection.
* **value\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the value projection.
* **proj\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the output projection.
* **mlp\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the MLP layers.
* **layerscale\_value** (`float`, *optional*, defaults to 1.0) —
  Initial value to use for layer scale.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  Stochastic depth rate per sample (when applied in the main path of residual layers).
* **use\_gated\_mlp** (`bool`, *optional*, defaults to `False`) —
  Whether to use the SwiGLU feedforward neural network.
* **num\_register\_tokens** (`int`, *optional*, defaults to 0) —
  The number of register tokens.
* **pos\_embed\_shift** (`float`, *optional*) —
  Amount to randomly shift position embedding coordinates in [-shift, shift],
  applied only in training mode if not `None`.
* **pos\_embed\_jitter** (`float`, *optional*) —
  Amount to randomly jitter position embedding coordinates in log-uniform value in [1/jitter, jitter],
  applied only in training mode if not `None`.
* **pos\_embed\_rescale** (`float`, *optional*, defaults to 2.0) —
  Amount to randomly rescale position embedding coordinates in log-uniform value in [1/rescale, rescale],
  applied only in training mode if not `None`.

This is the configuration class to store the configuration of a `DINOv3Model`. It is used to instantiate an
DINOv3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the DINOv3
[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DINOv3ViTConfig, DINOv3ViTModel

>>> # Initializing a DINOv3 ViT-small style configuration
>>> config = DINOv3ViTConfig()

>>> # Initializing a model (with random weights) from the config
>>> model = DINOv3ViTModel(config)

>>> # Accessing the model config
>>> config = model.config
```

## DINOv3ConvNextConfig

### class transformers.DINOv3ConvNextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_convnext/configuration_dinov3_convnext.py#L26)

( num\_channels: int = 3 hidden\_sizes: typing.Optional[list[int]] = None depths: typing.Optional[list[int]] = None hidden\_act: str = 'gelu' initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-06 layer\_scale\_init\_value: float = 1e-06 drop\_path\_rate: float = 0.0 image\_size: int = 224 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to [96, 192, 384, 768]) —
  Dimensionality (hidden size) at each stage.
* **depths** (`list[int]`, *optional*, defaults to [3, 3, 9, 3]) —
  The number of layers for each stage.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
  `"selu"` and `"gelu_new"` are supported.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **layer\_scale\_init\_value** (`float`, *optional*, defaults to 1e-06) —
  The initial value for the layer scale.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  The drop rate for stochastic depth.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of input images.

This is the configuration class to store the configuration of a [DINOv3ConvNextModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextModel). It is used to instantiate an
DINOv3ConvNext model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the DINOv3ConvNext
[facebook/dinov3-convnext-tiny-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

>>> # Initializing a DINOv3ConvNext (tiny variant) style configuration
>>> config = DINOv3ConvNextConfig()

>>> # Initializing a model (with random weights)
>>> model = DINOv3ConvNextModel(config)

>>> # Accessing the model config
>>> config = model.config
```

## DINOv3ViTModel

### class transformers.DINOv3ViTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py#L487)

( config: DINOv3ViTConfig  )

Parameters

* **config** ([DINOv3ViTConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dinov3 Vit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py#L502)

( pixel\_values: Tensor bool\_masked\_pos: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0). Only relevant for
  pre-training.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DINOv3ViTConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTConfig)) and inputs.

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

The [DINOv3ViTModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## DINOv3ConvNextModel

### class transformers.DINOv3ConvNextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_convnext/modeling_dinov3_convnext.py#L217)

( config: DINOv3ConvNextConfig  )

Parameters

* **config** ([DINOv3ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dinov3 Convnext Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_convnext/modeling_dinov3_convnext.py#L226)

( pixel\_values: FloatTensor output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DINOv3ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [DINOv3ConvNextModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## DINOv3ViTImageProcessorFast

### class transformers.DINOv3ViTImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_vit/image_processing_dinov3_vit_fast.py#L47)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Dinov3 Vit image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L639)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*args \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dinov3.md)
