*This model was released on 2025-06-11 and added to Hugging Face Transformers on 2025-06-11.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

# V-JEPA 2

[V-JEPA 2](https://huggingface.co/papers/2506.09985) ([blog post](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)) is a self-supervised approach to training video encoders developed by FAIR, Meta. Using internet-scale video data, V-JEPA 2 attains state-of-the-art performance on motion understanding and human action anticipation tasks. V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 (using a small amount of robot trajectory interaction data) that solves robot manipulation tasks without environment-specific data collection or task-specific training or calibration.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vjepa.gif)

You can find all original V-JEPA2 checkpoints under the [V-JEPA 2](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) collection.

This model was contributed by [koustuvs](https://huggingface.co/koustuvs), [yonigozlan](https://huggingface.co/yonigozlan) and [qubvel](https://huggingface.co/qubvel-hf). The original code can be found [here](https://github.com/facebookresearch/vjepa2).

## Usage example

The snippet below shows how to load the V-JEPA 2 model for feature extraction using the `AutoModel` class.


```
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="pt").to(model.device)
outputs = model(**video)

# V-JEPA 2 encoder outputs, same as calling `model.get_vision_features()`
encoder_outputs = outputs.last_hidden_state

# V-JEPA 2 predictor outputs
predictor_outputs = outputs.predictor_output.last_hidden_state
```

V-JEPA 2 can also be finetuned for video classification. In the following snippet, we show how use finetuned on Something-Something-V2 video classification model.


```
import torch
import numpy as np

from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification, infer_device

device = infer_device()

# Load model and video preprocessor
hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"

model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

# To load a video, sample the number of frames according to the model.
video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, model.config.frames_per_clip, 8) # you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width

# Preprocess and run inference
inputs = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names:")
top5_indices = logits.topk(5).indices[0]
top5_probs = torch.softmax(logits, dim=-1).topk(5).values[0]
for idx, prob in zip(top5_indices, top5_probs):
    text_label = model.config.id2label[idx.item()]
    print(f" - {text_label}: {prob:.2f}")
```

## VJEPA2Config

### class transformers.VJEPA2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vjepa2/configuration_vjepa2.py#L20)

( patch\_size = 16 crop\_size = 256 frames\_per\_clip = 64 tubelet\_size = 2 hidden\_size = 1024 in\_chans = 3 num\_attention\_heads = 16 num\_hidden\_layers = 24 drop\_path\_rate = 0.0 mlp\_ratio = 4.0 layer\_norm\_eps = 1e-06 qkv\_bias = True attention\_probs\_dropout\_prob = 0.0 hidden\_act = 'gelu' initializer\_range = 0.02 attention\_dropout = 0.0 num\_pooler\_layers = 3 pred\_hidden\_size = 384 pred\_num\_attention\_heads = 12 pred\_num\_hidden\_layers = 12 pred\_num\_mask\_tokens = 10 pred\_zero\_init\_mask\_tokens = True pred\_mlp\_ratio = 4.0 \*\*kwargs  )

Parameters

* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **crop\_size** (`int`, *optional*, defaults to 256) —
  Input resolution of the model
* **frames\_per\_clip** (`int`, *optional*, defaults to 64) —
  The number of frames the model has been pretrained with. Does not impact inference.
* **tubelet\_size** (`int`, *optional*, defaults to 2) —
  The number of temporal frames used for a single rastor, check paper for more information.
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers
* **in\_chans** (`int`, *optional*, defaults to 3) —
  The number of input channels
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Encoder
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  The number of hidden layers
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  Stochastic depth rate per sample (when applied in the main path of residual layers).
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  Ratio of the hidden size of the MLPs used in Encoder relative to the `hidden_size`.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for attentions.
  The dropout probability for all fully connected layers.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for attentions.
* **num\_pooler\_layers** (`int`, *optional*, defaults to 3) —
  The number of self-attention layers in the pooler.
* **pred\_hidden\_size** (`int`, *optional*, defaults to 384) —
  Dimensionality of the predictor layers
* **pred\_num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Predictor
* **pred\_num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Predictor
* **pred\_num\_mask\_tokens** (`int`, *optional*, defaults to 10) —
  Define the number of mask tokens to use in the Predictor
* **pred\_zero\_init\_mask\_tokens** (`bool`, *optional*, defaults to `True`) —
  Initialize the mask tokens in the predictor with 0.
* **pred\_mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  Ratio of the hidden size of the MLPs used in Predictor relative to the `pred_hidden_size`.

This is the configuration class to store the configuration of a [VJEPA2Model](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Model). It is used to instantiate an
VJEPA2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the VJEPA2
[facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import VJEPA2Config, VJEPA2Model

>>> # Initializing a VJEPA2 vjepa2-vitl-fpc64-256 style configuration
>>> configuration = VJEPA2Config()

>>> # Initializing a model (with random weights) from the vjepa2-vitl-fpc64-256  style configuration
>>> model = VJEPA2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VJEPA2Model

### class transformers.VJEPA2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vjepa2/modeling_vjepa2.py#L1047)

( config: VJEPA2Config  )

Parameters

* **config** ([VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Vjepa2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vjepa2/modeling_vjepa2.py#L1061)

( pixel\_values\_videos: Tensor context\_head\_mask: typing.Optional[torch.Tensor] = None context\_mask: typing.Optional[list[torch.Tensor]] = None target\_head\_mask: typing.Optional[torch.Tensor] = None target\_mask: typing.Optional[list[torch.Tensor]] = None skip\_predictor: bool = False output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.vjepa2.modeling_vjepa2.VJEPA2WithMaskedInputModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values\_videos** (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [VJEPA2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2VideoProcessor). See `VJEPA2VideoProcessor.__call__()` for details (`processor_class` uses
  [VJEPA2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2VideoProcessor) for processing videos).
* **context\_head\_mask** (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*) —
  The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the context.
* **context\_mask** (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*) —
  The mask position ids indicating which encoder output patches are going to be exposed to the predictor.
  By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating full context
  available to the predictor.
* **target\_head\_mask** (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*) —
  The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the target.
* **target\_mask** (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*) —
  The mask position ids indicating which encoder output patches are going to be used as a prediction target
  for the predictor. By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating
  that the predictor should predict all encoder patches.
* **skip\_predictor** (`bool`, defaults to `False`) —
  flag to skip the predictor forward, useful if you just need the encoder outputs
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.vjepa2.modeling_vjepa2.VJEPA2WithMaskedInputModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.vjepa2.modeling_vjepa2.VJEPA2WithMaskedInputModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config)) and inputs.

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **masked\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs) — The masked hidden state of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **predictor\_output** (`VJEPA2WithMaskedInputPredictorOutput`, *optional*) — The output from the Predictor module.

The [VJEPA2Model](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## VJEPA2ForVideoClassification

### class transformers.VJEPA2ForVideoClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vjepa2/modeling_vjepa2.py#L1155)

( config: VJEPA2Config  )

Parameters

* **config** ([VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

V-JEPA 2 Model transformer with a video classification head on top (a linear layer on top of the attentive pooler).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vjepa2/modeling_vjepa2.py#L1169)

( pixel\_values\_videos: Tensor labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values\_videos** (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [VJEPA2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2VideoProcessor). See `VJEPA2VideoProcessor.__call__()` for details (`processor_class` uses
  [VJEPA2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2VideoProcessor) for processing videos).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VJEPA2ForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2ForVideoClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> import numpy as np
>>> from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

>>> device = "cuda"

>>> video_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
>>> model = VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2").to(device)

>>> video = np.ones((64, 256, 256, 3))  # 64 frames, 256x256 RGB
>>> inputs = video_processor(video, return_tensors="pt").to(device)

>>> # For inference
>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> logits = outputs.logits

>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])

>>> # For training
>>> labels = torch.ones(1, dtype=torch.long, device=device)
>>> loss = model(**inputs, labels=labels).loss
```

## VJEPA2VideoProcessor

### class transformers.VJEPA2VideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vjepa2/video_processing_vjepa2.py#L35)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.vjepa2.video\_processing\_vjepa2.VJEPA2VideoProcessorInitKwargs]  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vjepa2.md)
