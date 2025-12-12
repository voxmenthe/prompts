*This model was released on 2021-10-12 and added to Hugging Face Transformers on 2021-10-26.*

# UniSpeech-SAT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The UniSpeech-SAT model was proposed in [UniSpeech-SAT: Universal Speech Representation Learning with Speaker Aware
Pre-Training](https://huggingface.co/papers/2110.05752) by Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen,
Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu .

The abstract from the paper is the following:

*Self-supervised learning (SSL) is a long-standing goal for speech processing, since it utilizes large-scale unlabeled
data and avoids extensive human labeling. Recent years witness great successes in applying self-supervised learning in
speech recognition, while limited exploration was attempted in applying SSL for modeling speaker characteristics. In
this paper, we aim to improve the existing SSL framework for speaker representation learning. Two methods are
introduced for enhancing the unsupervised speaker information extraction. First, we apply the multi-task learning to
the current SSL framework, where we integrate the utterance-wise contrastive loss with the SSL objective function.
Second, for better speaker discrimination, we propose an utterance mixing strategy for data augmentation, where
additional overlapped utterances are created unsupervisedly and incorporate during training. We integrate the proposed
methods into the HuBERT framework. Experiment results on SUPERB benchmark show that the proposed system achieves
state-of-the-art performance in universal representation learning, especially for speaker identification oriented
tasks. An ablation study is performed verifying the efficacy of each proposed method. Finally, we scale up training
dataset to 94 thousand hours public audio data and achieve further performance improvement in all SUPERB tasks.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors’ code can be
found [here](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT).

## Usage tips

* UniSpeechSat is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
  Please use [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) for the feature extraction.
* UniSpeechSat model can be fine-tuned using connectionist temporal classification (CTC) so the model output has to be
  decoded using [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).
* UniSpeechSat performs especially well on speaker verification, speaker identification, and speaker diarization tasks.

> [!NOTE]
> The `head_mask` argument is ignored when using all attention implementation other than “eager”. If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

## Resources

* [Audio classification task guide](../tasks/audio_classification)
* [Automatic speech recognition task guide](../tasks/asr)

## UniSpeechSatConfig

### class transformers.UniSpeechSatConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/configuration_unispeech_sat.py#L27)

( vocab\_size = 32 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout = 0.1 activation\_dropout = 0.1 attention\_dropout = 0.1 feat\_proj\_dropout = 0.0 feat\_quantizer\_dropout = 0.0 final\_dropout = 0.1 layerdrop = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 feat\_extract\_norm = 'group' feat\_extract\_activation = 'gelu' conv\_dim = (512, 512, 512, 512, 512, 512, 512) conv\_stride = (5, 2, 2, 2, 2, 2, 2) conv\_kernel = (10, 3, 3, 3, 3, 2, 2) conv\_bias = False num\_conv\_pos\_embeddings = 128 num\_conv\_pos\_embedding\_groups = 16 do\_stable\_layer\_norm = False apply\_spec\_augment = True mask\_time\_prob = 0.05 mask\_time\_length = 10 mask\_time\_min\_masks = 2 mask\_feature\_prob = 0.0 mask\_feature\_length = 10 mask\_feature\_min\_masks = 0 num\_codevectors\_per\_group = 320 num\_codevector\_groups = 2 contrastive\_logits\_temperature = 0.1 num\_negatives = 100 codevector\_dim = 256 proj\_codevector\_dim = 256 diversity\_loss\_weight = 0.1 ctc\_loss\_reduction = 'mean' ctc\_zero\_infinity = False use\_weighted\_layer\_sum = False classifier\_proj\_size = 256 tdnn\_dim = (512, 512, 512, 512, 1500) tdnn\_kernel = (5, 3, 3, 1, 1) tdnn\_dilation = (1, 2, 3, 1, 1) xvector\_output\_dim = 512 pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 num\_clusters = 504 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32) —
  Vocabulary size of the UniSpeechSat model. Defines the number of different tokens that can be represented
  by the `inputs_ids` passed when calling [UniSpeechSatModel](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel). Vocabulary size of the model. Defines the
  different tokens that can be represented by the *inputs\_ids* passed to the forward method of
  [UniSpeechSatModel](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel).
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
* **hidden\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for activations inside the fully connected layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **feat\_proj\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for output of the feature encoder.
* **feat\_quantizer\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for the output of the feature encoder that’s used by the quantizer.
* **final\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the final projection layer of [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC).
* **layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>) for more
  details.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **feat\_extract\_norm** (`str`, *optional*, defaults to `"group"`) —
  The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
  normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
  convolutional layers.
* **feat\_extract\_activation** (`str, *optional*, defaults to` “gelu”`) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string,` “gelu”`,` “relu”`,` “selu”`and`“gelu\_new”` are supported.
* **conv\_dim** (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`) —
  A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
  feature encoder. The length of *conv\_dim* defines the number of 1D convolutional layers.
* **conv\_stride** (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`) —
  A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
  of *conv\_stride* defines the number of convolutional layers and has to match the length of *conv\_dim*.
* **conv\_kernel** (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
  length of *conv\_kernel* defines the number of convolutional layers and has to match the length of
  *conv\_dim*.
* **conv\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether the 1D convolutional layers have a bias.
* **num\_conv\_pos\_embeddings** (`int`, *optional*, defaults to 128) —
  Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
  embeddings layer.
* **num\_conv\_pos\_embedding\_groups** (`int`, *optional*, defaults to 16) —
  Number of groups of 1D convolutional positional embeddings layer.
* **do\_stable\_layer\_norm** (`bool`, *optional*, defaults to `False`) —
  Whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is False` corresponds to applying layer norm after the attention layer.
* **apply\_spec\_augment** (`bool`, *optional*, defaults to `True`) —
  Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
  [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
  Recognition](https://huggingface.co/papers/1904.08779).
* **mask\_time\_prob** (`float`, *optional*, defaults to 0.05) —
  Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
  procedure generates ”mask\_time\_prob*len(time\_axis)/mask\_time\_length” independent masks over the axis. If
  reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
  masked,* mask\_time\_prob *should be `prob\_vector\_start*mask\_time\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is True`.
* **mask\_time\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the time axis.
* **mask\_time\_min\_masks** (`int`, *optional*, defaults to 2) —
  The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
  irrespectively of `mask_feature_prob`. Only relevant if ”mask\_time\_prob\*len(time\_axis)/mask\_time\_length <
  mask\_time\_min\_masks”
* **mask\_feature\_prob** (`float`, *optional*, defaults to 0.0) —
  Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
  masking procedure generates ”mask\_feature\_prob*len(feature\_axis)/mask\_time\_length” independent masks over
  the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
  span to be masked,* mask\_feature\_prob *should be `prob\_vector\_start*mask\_feature\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is
  True`.
* **mask\_feature\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the feature axis.
* **mask\_feature\_min\_masks** (`int`, *optional*, defaults to 0) —
  The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
  step, irrespectively of `mask_feature_prob`. Only relevant if
  ”mask\_feature\_prob\*len(feature\_axis)/mask\_feature\_length < mask\_feature\_min\_masks”
* **num\_codevectors\_per\_group** (`int`, *optional*, defaults to 320) —
  Number of entries in each quantization codebook (group).
* **num\_codevector\_groups** (`int`, *optional*, defaults to 2) —
  Number of codevector groups for product codevector quantization.
* **contrastive\_logits\_temperature** (`float`, *optional*, defaults to 0.1) —
  The temperature *kappa* in the contrastive loss.
* **num\_negatives** (`int`, *optional*, defaults to 100) —
  Number of negative samples for the contrastive loss.
* **codevector\_dim** (`int`, *optional*, defaults to 256) —
  Dimensionality of the quantized feature vectors.
* **proj\_codevector\_dim** (`int`, *optional*, defaults to 256) —
  Dimensionality of the final projection of both the quantized and the transformer features.
* **diversity\_loss\_weight** (`int`, *optional*, defaults to 0.1) —
  The weight of the codebook diversity loss component.
* **ctc\_loss\_reduction** (`str`, *optional*, defaults to `"mean"`) —
  Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
  instance of [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC).
* **ctc\_zero\_infinity** (`bool`, *optional*, defaults to `False`) —
  Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
  occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
  of [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC).
* **use\_weighted\_layer\_sum** (`bool`, *optional*, defaults to `False`) —
  Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
  instance of [UniSpeechSatForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForSequenceClassification).
* **classifier\_proj\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the projection before token mean-pooling for classification.
* **tdnn\_dim** (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 1500)`) —
  A tuple of integers defining the number of output channels of each 1D convolutional layer in the *TDNN*
  module of the *XVector* model. The length of *tdnn\_dim* defines the number of *TDNN* layers.
* **tdnn\_kernel** (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 3, 3, 1, 1)`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the *TDNN* module of the
  *XVector* model. The length of *tdnn\_kernel* has to match the length of *tdnn\_dim*.
* **tdnn\_dilation** (`tuple[int]` or `list[int]`, *optional*, defaults to `(1, 2, 3, 1, 1)`) —
  A tuple of integers defining the dilation factor of each 1D convolutional layer in *TDNN* module of the
  *XVector* model. The length of *tdnn\_dilation* has to match the length of *tdnn\_dim*.
* **xvector\_output\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of the *XVector* embedding vectors.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the padding token.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the “beginning-of-sequence” token.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the “end-of-sequence” token.
* **num\_clusters** (`int`, *optional*, defaults to 504) —
  Number of clusters for weak labeling. Only relevant when using an instance of
  [UniSpeechSatForPreTraining](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForPreTraining).

This is the configuration class to store the configuration of a [UniSpeechSatModel](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel). It is used to instantiate an
UniSpeechSat model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the UniSpeechSat
[microsoft/unispeech-sat-base-100h-libri-ft](https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import UniSpeechSatModel, UniSpeechSatConfig

>>> # Initializing a UniSpeechSat microsoft/unispeech-sat-base-100h-libri-ft style configuration
>>> configuration = UniSpeechSatConfig()

>>> # Initializing a model from the microsoft/unispeech-sat-base-100h-libri-ft style configuration
>>> model = UniSpeechSatModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## UniSpeechSat specific outputs

### class transformers.models.unispeech\_sat.modeling\_unispeech\_sat.UniSpeechSatForPreTrainingOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L66)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None projected\_states: typing.Optional[torch.FloatTensor] = None projected\_quantized\_states: typing.Optional[torch.FloatTensor] = None codevector\_perplexity: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`*optional*`, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`) —
  Total loss as the sum of the contrastive loss (L\_m) and the diversity loss (L\_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`, *optional*) —
  Prediction scores of the contrastive loss model, i.e. the output of the model before the final softmax.
* **projected\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) —
  Hidden-states of the model projected to *config.proj\_codevector\_dim* that can be used to predict the masked
  projected quantized states.
* **projected\_quantized\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) —
  Quantized extracted feature vectors projected to *config.proj\_codevector\_dim* representing the positive
  target vectors for contrastive loss.
* **codevector\_perplexity** (`torch.FloatTensor` of shape `(1,)`) —
  The perplexity of the codevector distribution, used to measure the diversity of the codebook.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of `UniSpeechSatForPreTrainingOutput`, with potential hidden states and attentions.

## UniSpeechSatModel

### class transformers.UniSpeechSatModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L982)

( config: UniSpeechSatConfig  )

Parameters

* **config** ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Unispeech Sat Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1045)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None mask\_time\_indices: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mask\_time\_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
  masked extracted features in *config.proj\_codevector\_dim* space.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **extract\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) — Sequence of extracted feature vectors of the last convolutional layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [UniSpeechSatModel](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## UniSpeechSatForCTC

### class transformers.UniSpeechSatForCTC

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1236)

( config target\_lang: typing.Optional[str] = None  )

Parameters

* **config** ([UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **target\_lang** (`str`, *optional*) —
  Language id of adapter weights. Adapter weights are stored in the format adapter..safetensors or
  adapter..bin. Only relevant when using an instance of [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC) with adapters. Uses ‘eng’ by
  default.

UniSpeechSat Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1314)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*) —
  Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
  the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
  All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`.

Returns

[transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, UniSpeechSatForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
>>> model = UniSpeechSatForCTC.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
...

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
...
```

## UniSpeechSatForSequenceClassification

### class transformers.UniSpeechSatForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1392)

( config  )

Parameters

* **config** ([UniSpeechSatForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UniSpeechSat Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1437)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See `UniSpeechSatProcessor.__call__` for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [UniSpeechSatForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, UniSpeechSatForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
>>> model = UniSpeechSatForSequenceClassification.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = UniSpeechSatForSequenceClassification.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, UniSpeechSatForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
>>> model = UniSpeechSatForSequenceClassification.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = UniSpeechSatForSequenceClassification.from_pretrained(
...     "microsoft/unispeech-sat-base-100h-libri-ft", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## UniSpeechSatForAudioFrameClassification

### class transformers.UniSpeechSatForAudioFrameClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1508)

( config  )

Parameters

* **config** ([UniSpeechSatForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForAudioFrameClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Unispeech Sat Model with a frame classification head on top for tasks like Speaker Diarization.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1552)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See `UniSpeechSatProcessor.__call__` for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [UniSpeechSatForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoFeatureExtractor, UniSpeechSatForAudioFrameClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
>>> model = UniSpeechSatForAudioFrameClassification.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")

>>> # audio file is decoded on the fly
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> probabilities = torch.sigmoid(logits[0])
>>> # labels is a one-hot array of shape (num_frames, num_speakers)
>>> labels = (probabilities > 0.5).long()
>>> labels[0].tolist()
...
```

## UniSpeechSatForXVector

### class transformers.UniSpeechSatForXVector

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1673)

( config  )

Parameters

* **config** ([UniSpeechSatForXVector](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UniSpeechSat Model with an XVector feature extraction head on top for tasks like Speaker Verification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1735)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See `UniSpeechSatProcessor.__call__` for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) — Classification hidden states before AMSoftmax.
* **embeddings** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) — Utterance embeddings used for vector similarity-based retrieval.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [UniSpeechSatForXVector](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoFeatureExtractor, UniSpeechSatForXVector
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
>>> model = UniSpeechSatForXVector.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")

>>> # audio file is decoded on the fly
>>> inputs = feature_extractor(
...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
... )
>>> with torch.no_grad():
...     embeddings = model(**inputs).embeddings

>>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

>>> # the resulting embeddings can be used for cosine similarity-based retrieval
>>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
>>> similarity = cosine_sim(embeddings[0], embeddings[1])
>>> threshold = 0.7  # the optimal threshold is dataset-dependent
>>> if similarity < threshold:
...     print("Speakers are not the same!")
>>> round(similarity.item(), 2)
...
```

## UniSpeechSatForPreTraining

### class transformers.UniSpeechSatForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1104)

( config: UniSpeechSatConfig  )

Parameters

* **config** ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UniSpeechSat Model with a vector-quantization module and ctc loss for pre-training.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py#L1172)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.unispeech\_sat.modeling\_unispeech\_sat.UniSpeechSatForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.unispeech\_sat.modeling\_unispeech\_sat.UniSpeechSatForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.unispeech\_sat.modeling\_unispeech\_sat.UniSpeechSatForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig)) and inputs.

* **loss** (`*optional*`, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the contrastive loss (L\_m) and the diversity loss (L\_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`, *optional*) — Prediction scores of the contrastive loss model, i.e. the output of the model before the final softmax.
* **projected\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) — Hidden-states of the model projected to *config.proj\_codevector\_dim* that can be used to predict the masked
  projected quantized states.
* **projected\_quantized\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) — Quantized extracted feature vectors projected to *config.proj\_codevector\_dim* representing the positive
  target vectors for contrastive loss.
* **codevector\_perplexity** (`torch.FloatTensor` of shape `(1,)`) — The perplexity of the codevector distribution, used to measure the diversity of the codebook.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [UniSpeechSatForPreTraining](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoFeatureExtractor, UniSpeechSatForPreTraining
>>> from transformers.models.unispeech_sat.modeling_unispeech_sat import _compute_mask_indices

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
>>> model = UniSpeechSatForPreTraining.from_pretrained("microsoft/unispeech-sat-base")
>>> # TODO: Add full pretraining example
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/unispeech-sat.md)
