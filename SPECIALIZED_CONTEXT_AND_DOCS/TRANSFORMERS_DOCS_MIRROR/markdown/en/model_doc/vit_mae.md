*This model was released on 2021-11-11 and added to Hugging Face Transformers on 2022-01-18.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# ViTMAE

[ViTMAE](https://huggingface.co/papers/2111.06377) is a self-supervised vision model that is pretrained by masking large portions of an image (~75%). An encoder processes the visible image patches and a decoder reconstructs the missing pixels from the encoded patches and mask tokens. After pretraining, the encoder can be reused for downstream tasks like image classification or object detection — often outperforming models trained with supervised learning.

![drawing](https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png)

You can find all the original ViTMAE checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=vit-mae) organization.

Click on the ViTMAE models in the right sidebar for more examples of how to apply ViTMAE to vision tasks.

The example below demonstrates how to reconstruct the missing pixels with the [ViTMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) class.

AutoModel


```
import torch
import requests
from PIL import Image
from transformers import infer_device, ViTImageProcessor, ViTMAEForPreTraining

device = infer_device()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
inputs = processor(image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa").to(device)
with torch.no_grad():
    outputs = model(**inputs)

reconstruction = outputs.logits
```

## Notes

* ViTMAE is typically used in two stages. Self-supervised pretraining with [ViTMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining), and then discarding the decoder and fine-tuning the encoder. After fine-tuning, the weights can be plugged into a model like [ViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForImageClassification).
* Use [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for input preparation.

## Resources

* Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb) to learn how to visualize the reconstructed pixels from [ViTMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining).

## ViTMAEConfig

### class transformers.ViTMAEConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_mae/configuration_vit_mae.py#L24)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 image\_size = 224 patch\_size = 16 num\_channels = 3 qkv\_bias = True decoder\_num\_attention\_heads = 16 decoder\_hidden\_size = 512 decoder\_num\_hidden\_layers = 8 decoder\_intermediate\_size = 2048 mask\_ratio = 0.75 norm\_pix\_loss = False \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **decoder\_num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the decoder.
* **decoder\_hidden\_size** (`int`, *optional*, defaults to 512) —
  Dimensionality of the decoder.
* **decoder\_num\_hidden\_layers** (`int`, *optional*, defaults to 8) —
  Number of hidden layers in the decoder.
* **decoder\_intermediate\_size** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the decoder.
* **mask\_ratio** (`float`, *optional*, defaults to 0.75) —
  The ratio of the number of masked tokens in the input sequence.
* **norm\_pix\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether or not to train with normalized pixels (see Table 3 in the paper). Using normalized pixels improved
  representation quality in the experiments of the authors.

This is the configuration class to store the configuration of a [ViTMAEModel](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEModel). It is used to instantiate an ViT
MAE model according to the specified arguments, defining the model architecture. Instantiating a configuration with
the defaults will yield a similar configuration to that of the ViT
[facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ViTMAEConfig, ViTMAEModel

>>> # Initializing a ViT MAE vit-mae-base style configuration
>>> configuration = ViTMAEConfig()

>>> # Initializing a model (with random weights) from the vit-mae-base style configuration
>>> model = ViTMAEModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ViTMAEModel

### class transformers.ViTMAEModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_mae/modeling_vit_mae.py#L575)

( config  )

Parameters

* **config** ([ViTMAEModel](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Vit Mae Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_mae/modeling_vit_mae.py#L599)

( pixel\_values: typing.Optional[torch.FloatTensor] = None noise: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None interpolate\_pos\_encoding: bool = False \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.vit_mae.modeling_vit_mae.ViTMAEModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **noise** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mainly used for testing purposes to control randomness and maintain the reproducibility
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **interpolate\_pos\_encoding** (`bool`, *optional*, default `False`) —
  Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
  resolution images.

Returns

`transformers.models.vit_mae.modeling_vit_mae.ViTMAEModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.vit_mae.modeling_vit_mae.ViTMAEModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTMAEConfig](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Tensor indicating which patches are masked (1) and which are not (0).
* **ids\_restore** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Tensor containing the original index of the (shuffled) masked patches.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ViTMAEModel](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ViTMAEModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ViTMAEForPreTraining

### class transformers.ViTMAEForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_mae/modeling_vit_mae.py#L777)

( config: ViTMAEConfig  )

Parameters

* **config** ([ViTMAEConfig](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.

Note that we provide a script to pre-train this model on custom data in our [examples
directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_mae/modeling_vit_mae.py#L907)

( pixel\_values: typing.Optional[torch.FloatTensor] = None noise: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None interpolate\_pos\_encoding: bool = False \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.vit_mae.modeling_vit_mae.ViTMAEForPreTrainingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **noise** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mainly used for testing purposes to control randomness and maintain the reproducibility
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **interpolate\_pos\_encoding** (`bool`, *optional*, default `False`) —
  Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
  resolution images.

Returns

`transformers.models.vit_mae.modeling_vit_mae.ViTMAEForPreTrainingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.vit_mae.modeling_vit_mae.ViTMAEForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTMAEConfig](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`) — Pixel reconstruction loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`) — Pixel reconstruction logits.
* **mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Tensor indicating which patches are masked (1) and which are not (0).
* **ids\_restore** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Tensor containing the original index of the (shuffled) masked patches.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ViTMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ViTMAEForPreTraining
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> loss = outputs.loss
>>> mask = outputs.mask
>>> ids_restore = outputs.ids_restore
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vit_mae.md)
