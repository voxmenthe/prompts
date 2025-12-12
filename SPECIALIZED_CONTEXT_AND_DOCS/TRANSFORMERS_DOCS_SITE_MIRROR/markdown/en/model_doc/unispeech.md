# UniSpeech

## Overview

The UniSpeech model was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://huggingface.co/papers/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael
Zeng, Xuedong Huang .

The abstract from the paper is the following:

*In this paper, we propose a unified pre-training approach called UniSpeech to learn speech representations with both
unlabeled and labeled data, in which supervised phonetic CTC learning and phonetically-aware contrastive
self-supervised learning are conducted in a multi-task learning manner. The resultant representations can capture
information more correlated with phonetic structures and improve the generalization across languages and domains. We
evaluate the effectiveness of UniSpeech for cross-lingual representation learning on public CommonVoice corpus. The
results show that UniSpeech outperforms self-supervised pretraining and supervised transfer learning for speech
recognition by a maximum of 13.4% and 17.8% relative phone error rate reductions respectively (averaged over all
testing languages). The transferability of UniSpeech is also demonstrated on a domain-shift speech recognition task,
i.e., a relative word error rate reduction of 6% against the previous approach.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors' code can be
found [here](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech).

## Usage tips

- UniSpeech is a speech model that accepts a float array corresponding to the raw waveform of the speech signal. Please
  use [Wav2Vec2Processor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) for the feature extraction.
- UniSpeech model can be fine-tuned using connectionist temporal classification (CTC) so the model output has to be
  decoded using [Wav2Vec2CTCTokenizer](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

## Resources

- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

## UniSpeechConfig[[transformers.UniSpeechConfig]]

#### transformers.UniSpeechConfig[[transformers.UniSpeechConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/unispeech/configuration_unispeech.py#L27)

This is the configuration class to store the configuration of a [UniSpeechModel](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechModel). It is used to instantiate an
UniSpeech model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the UniSpeech
[microsoft/unispeech-large-1500h-cv](https://huggingface.co/microsoft/unispeech-large-1500h-cv) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import UniSpeechConfig, UniSpeechModel

>>> # Initializing a UniSpeech facebook/unispeech-base-960h style configuration
>>> configuration = UniSpeechConfig()

>>> # Initializing a model (with random weights) from the facebook/unispeech-base-960h style configuration
>>> model = UniSpeechModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 32) : Vocabulary size of the UniSpeech model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [UniSpeechModel](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechModel). Vocabulary size of the model. Defines the different tokens that can be represented by the *inputs_ids* passed to the forward method of [UniSpeechModel](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechModel).

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

activation_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for activations inside the fully connected layer.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

feat_proj_dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for output of the feature encoder.

feat_quantizer_dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for the output of the feature encoder that's used by the quantizer.

final_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for the final projection layer of [UniSpeechForCTC](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechForCTC).

layerdrop (`float`, *optional*, defaults to 0.1) : The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

feat_extract_norm (`str`, *optional*, defaults to `"group"`) : The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D convolutional layers.

feat_extract_activation (`str, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`) : A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.

conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`) : A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.

conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`) : A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The length of *conv_kernel* defines the number of convolutional layers and has to match the length of *conv_dim*.

conv_bias (`bool`, *optional*, defaults to `False`) : Whether the 1D convolutional layers have a bias.

num_conv_pos_embeddings (`int`, *optional*, defaults to 128) : Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional embeddings layer.

num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16) : Number of groups of 1D convolutional positional embeddings layer.

do_stable_layer_norm (`bool`, *optional*, defaults to `False`) : Whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is False` corresponds to applying layer norm after the attention layer.

apply_spec_augment (`bool`, *optional*, defaults to `True`) : Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://huggingface.co/papers/1904.08779).

mask_time_prob (`float`, *optional*, defaults to 0.05) : Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking procedure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector span to be masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.

mask_time_length (`int`, *optional*, defaults to 10) : Length of vector span along the time axis.

mask_time_min_masks (`int`, *optional*, defaults to 2) : The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step, irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length >> from transformers import AutoProcessor, UniSpeechForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> model = UniSpeechForCTC.from_pretrained("microsoft/unispeech-large-1500h-cv")

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

**Parameters:**

config ([UniSpeechForCTC](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechForCTC)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

target_lang (`str`, *optional*) : Language id of adapter weights. Adapter weights are stored in the format adapter..safetensors or adapter..bin. Only relevant when using an instance of [UniSpeechForCTC](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechForCTC) with adapters. Uses 'eng' by default.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechConfig](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## UniSpeechForSequenceClassification[[transformers.UniSpeechForSequenceClassification]]

#### transformers.UniSpeechForSequenceClassification[[transformers.UniSpeechForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/unispeech/modeling_unispeech.py#L1330)

UniSpeech Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.UniSpeechForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/unispeech/modeling_unispeech.py#L1363[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See `UniSpeechProcessor.__call__` for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechConfig](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [UniSpeechForSequenceClassification](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, UniSpeechForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> model = UniSpeechForSequenceClassification.from_pretrained("microsoft/unispeech-large-1500h-cv")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = UniSpeechForSequenceClassification.from_pretrained("microsoft/unispeech-large-1500h-cv", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, UniSpeechForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> model = UniSpeechForSequenceClassification.from_pretrained("microsoft/unispeech-large-1500h-cv", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = UniSpeechForSequenceClassification.from_pretrained(
...     "microsoft/unispeech-large-1500h-cv", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

**Parameters:**

config ([UniSpeechForSequenceClassification](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechConfig](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## UniSpeechForPreTraining[[transformers.UniSpeechForPreTraining]]

#### transformers.UniSpeechForPreTraining[[transformers.UniSpeechForPreTraining]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/unispeech/modeling_unispeech.py#L1055)

UniSpeech Model with a vector-quantization module and ctc loss for pre-training.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.UniSpeechForPreTraining.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/unispeech/modeling_unispeech.py#L1104[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput](/docs/transformers/main/en/model_doc/unispeech#transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput) or `tuple(torch.FloatTensor)`A [transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput](/docs/transformers/main/en/model_doc/unispeech#transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechConfig](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechConfig)) and inputs.

- **loss** (`*optional*`, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
- **projected_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
  projected quantized states.
- **projected_quantized_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
  target vectors for contrastive loss.
- **codevector_perplexity** (`torch.FloatTensor` of shape `(1,)`) -- The perplexity of the codevector distribution, used to measure the diversity of the codebook.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [UniSpeechForPreTraining](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoFeatureExtractor, UniSpeechForPreTraining

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> model = UniSpeechForPreTraining.from_pretrained("microsoft/unispeech-large-1500h-cv")
>>> # TODO: Add full pretraining example
```

**Parameters:**

config ([UniSpeechConfig](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput](/docs/transformers/main/en/model_doc/unispeech#transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput](/docs/transformers/main/en/model_doc/unispeech#transformers.models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UniSpeechConfig](/docs/transformers/main/en/model_doc/unispeech#transformers.UniSpeechConfig)) and inputs.

- **loss** (`*optional*`, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
- **projected_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
  projected quantized states.
- **projected_quantized_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
  target vectors for contrastive loss.
- **codevector_perplexity** (`torch.FloatTensor` of shape `(1,)`) -- The perplexity of the codevector distribution, used to measure the diversity of the codebook.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
