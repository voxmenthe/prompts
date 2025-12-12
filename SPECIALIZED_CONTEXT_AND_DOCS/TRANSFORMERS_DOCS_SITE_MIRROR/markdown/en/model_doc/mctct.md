# M-CTC-T

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

## Overview

The M-CTC-T model was proposed in [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://huggingface.co/papers/2111.00161) by Loren Lugosch, Tatiana Likhomanenko, Gabriel Synnaeve, and Ronan Collobert. The model is a 1B-param transformer encoder, with a CTC head over 8065 character labels and a language identification head over 60 language ID labels. It is trained on Common Voice (version 6.1, December 2020 release) and VoxPopuli. After training on Common Voice and VoxPopuli, the model is trained on Common Voice only. The labels are unnormalized character-level transcripts (punctuation and capitalization are not removed). The model takes as input Mel filterbank features from a 16Khz audio signal.

The abstract from the paper is the following:

*Semi-supervised learning through pseudo-labeling has become a staple of state-of-the-art monolingual
speech recognition systems. In this work, we extend pseudo-labeling to massively multilingual speech
recognition with 60 languages. We propose a simple pseudo-labeling recipe that works well even
with low-resource languages: train a supervised multilingual model, fine-tune it with semi-supervised
learning on a target language, generate pseudo-labels for that language, and train a final model using
pseudo-labels for all languages, either from scratch or by fine-tuning. Experiments on the labeled
Common Voice and unlabeled VoxPopuli datasets show that our recipe can yield a model with better
performance for many languages that also transfers well to LibriSpeech.*

This model was contributed by [cwkeam](https://huggingface.co/cwkeam). The original code can be found [here](https://github.com/flashlight/wav2letter/tree/main/recipes/mling_pl).

## Usage tips

The PyTorch version of this model is only available in torch 1.9 and higher.

## Resources

- [Automatic speech recognition task guide](../tasks/asr)

## MCTCTConfig[[transformers.MCTCTConfig]]

#### transformers.MCTCTConfig[[transformers.MCTCTConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/configuration_mctct.py#L24)

This is the configuration class to store the configuration of a [MCTCTModel](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTModel). It is used to instantiate an
M-CTC-T model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the M-CTC-T
[speechbrain/m-ctc-t-large](https://huggingface.co/speechbrain/m-ctc-t-large) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MCTCTConfig, MCTCTModel

>>> # Initializing a M-CTC-T mctct-large style configuration
>>> configuration = MCTCTConfig()

>>> # Initializing a model (with random weights) from the mctct-large style configuration
>>> model = MCTCTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 8065) : Vocabulary size of the M-CTC-T model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [MCTCTModel](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTModel).

hidden_size (`int`, *optional*, defaults to 1536) : Dimension of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 36) : Number of hidden layers in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 6144) : Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 4) : Number of attention heads for each attention layer in the Transformer encoder.

attention_head_dim (`int`, *optional*, defaults to 384) : Dimensions of each attention head for each attention layer in the Transformer encoder.

max_position_embeddings (`int`, *optional*, defaults to 920) : The maximum sequence length that this model might ever be used with (after log-mel spectrogram extraction).

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

layerdrop (`float`, *optional*, defaults to 0.3) : The probability of dropping an encoder layer during training. The default 0.3 value is used in the original implementation.

hidden_act (`str` or `function`, *optional*, defaults to `"relu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

hidden_dropout_prob (`float`, *optional*, defaults to 0.3) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.3) : The dropout ratio for the attention probabilities.

pad_token_id (`int`, *optional*, defaults to 1) : The tokenizer index of the pad token.

bos_token_id (`int`, *optional*, defaults to 0) : The tokenizer index of the bos token.

eos_token_id (`int`, *optional*, defaults to 2) : The tokenizer index of the eos token.

conv_glu_dim (`int`, *optional*, defaults to 1) : The dimension of the output of the `Conv1dSubsampler` layer in which GLU is applied on. Though the original Flashlight code uses the value of 2, here it's adapted to 1 due to transposition differences.

conv_dropout (`int`, *optional*, defaults to 0.3) : The probability of randomly dropping the `Conv1dSubsampler` layer during training.

num_conv_layers (`int`, *optional*, defaults to 1) : Number of convolution layers before applying transformer encoder layers.

conv_kernel (`Sequence[int]`, *optional*, defaults to `(7,)`) : The kernel size of the 1D convolution applied before transformer layers. `len(conv_kernel)` must be equal to `num_conv_layers`.

conv_stride (`Sequence[int]`, *optional*, defaults to `(3,)`) : The stride length of the 1D convolution applied before transformer layers. `len(conv_stride)` must be equal to `num_conv_layers`.

input_feat_per_channel (`int`, *optional*, defaults to 80) : Feature dimensions of the channels of the input to the Conv1D layer.

input_channels (`int`, *optional*, defaults to 1) : Number of input channels of the input to the Conv1D layer.

conv_channels (`list[int]`, *optional*) : Channel sizes of intermediate Conv1D layers.

ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`) : Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an instance of [MCTCTForCTC](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTForCTC).

ctc_zero_infinity (`bool`, *optional*, defaults to `False`) : Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance of [MCTCTForCTC](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTForCTC).

## MCTCTFeatureExtractor[[transformers.MCTCTFeatureExtractor]]

#### transformers.MCTCTFeatureExtractor[[transformers.MCTCTFeatureExtractor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/feature_extraction_mctct.py#L33)

Constructs a M-CTC-T feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods. This
code has been adapted from Flashlight's C++ code. For more information about the implementation, one can refer to
this [notebook](https://colab.research.google.com/drive/1GLtINkkhzms-IsdcGy_-tVCkv0qNF-Gt#scrollTo=pMCRGMmUC_an)
that takes the user step-by-step in the implementation.

__call__transformers.MCTCTFeatureExtractor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/feature_extraction_mctct.py#L161[{"name": "raw_speech", "val": ": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]]"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "truncation", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "sampling_rate", "val": ": typing.Optional[int] = None"}, {"name": "**kwargs", "val": ""}]- **raw_speech** (`torch.Tensor`, `np.ndarray`, `list[float]`, `list[torch.Tensor]`, `list[np.ndarray]`, `list[list[float]]`) --
  The sequence or batch of sequences to be padded. Each sequence can be a tensor, a numpy array, a list
  of float values, a list of tensors, a list of numpy arrays or a list of list of float values. Must be
  mono channel audio, not stereo, i.e. single float per timestep.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Select a strategy to pad the returned sequences (according to the model's padding side and padding
  index) among:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **max_length** (`int`, *optional*) --
  Maximum length of the returned list and optionally padding length (see above).
- **truncation** (`bool`) --
  Activates truncation to cut input sequences longer than *max_length* to *max_length*.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature_extractor's default.

  [What are attention masks?](../glossary#attention-mask)

- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.
- **sampling_rate** (`int`, *optional*) --
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.
- **padding_value** (`float`, defaults to 0.0) --0

Main method to featurize and prepare for the model one or several sequence(s). sequences. It returns the
log-mel spectrogram of the input audio, as implemented in the original Flashlight MFSC feature extraction code.

**Parameters:**

feature_size (`int`, defaults to 80) : The feature dimension of the extracted features. This is the number of mel_frequency

sampling_rate (`int`, defaults to 16000) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

padding_value (`float`, defaults to 0.0) : The value that is used to fill the padding values.

hop_length (`int`, defaults to 10) : Number of audio samples between windows. Otherwise referred to as "shift" in many papers.

win_length (`int`, defaults to 25) : Number of ms per window

win_function (`str`, defaults to `"hamming_window"`) : Name for the window function used for windowing, must be accessible via `torch.{win_function}`

frame_signal_scale (`float`, defaults to 32768.0) : Constant multiplied in creating the frames before applying DFT.

preemphasis_coeff (`float`, defaults to 0.97) : Constant multiplied in applying Pre-emphasis before DFT.

mel_floor (`float` defaults to 1.0) : Minimum value of mel frequency banks.

normalize_means (`bool`, *optional*, defaults to `True`) : Whether or not to zero-mean normalize the extracted features.

normalize_vars (`bool`, *optional*, defaults to `True`) : Whether or not to unit-variance normalize the extracted features.

## MCTCTProcessor[[transformers.MCTCTProcessor]]

#### transformers.MCTCTProcessor[[transformers.MCTCTProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/processing_mctct.py#L25)

Constructs a MCTCT processor which wraps a MCTCT feature extractor and a MCTCT tokenizer into a single processor.

[MCTCTProcessor](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTProcessor) offers all the functionalities of [MCTCTFeatureExtractor](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTFeatureExtractor) and [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See the
[__call__()](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTProcessor.__call__) and [decode()](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTProcessor.decode) for more information.

__call__transformers.MCTCTProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/processing_mctct.py#L47[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]

When used in normal mode, this method forwards all its arguments to MCTCTFeatureExtractor's
[__call__()](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTFeatureExtractor.__call__) and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to AutoTokenizer's
`__call__()`. Please refer to the docstring of the above two methods for more information.

**Parameters:**

feature_extractor (`MCTCTFeatureExtractor`) : An instance of [MCTCTFeatureExtractor](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTFeatureExtractor). The feature extractor is a required input.

tokenizer (`AutoTokenizer`) : An instance of [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). The tokenizer is a required input.
#### from_pretrained[[transformers.MCTCTProcessor.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1360)

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
#### save_pretrained[[transformers.MCTCTProcessor.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L782)

Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
can be reloaded using the [from_pretrained()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### batch_decode[[transformers.MCTCTProcessor.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1499)

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.
#### decode[[transformers.MCTCTProcessor.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/processing_mctct.py#L115)

This method forwards all its arguments to AutoTokenizer's [decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to the
docstring of this method for more information.

## MCTCTModel[[transformers.MCTCTModel]]

#### transformers.MCTCTModel[[transformers.MCTCTModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/modeling_mctct.py#L558)

The bare M-CTC-T Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.MCTCTModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/modeling_mctct.py#L568[{"name": "input_features", "val": ": Tensor"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_features** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [Wav2Vec2CTCTokenizer](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
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
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MCTCTConfig](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MCTCTModel](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoProcessor, MCTCTModel
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("speechbrain/m-ctc-t-large")
>>> model = MCTCTModel.from_pretrained("speechbrain/m-ctc-t-large")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
[1, 195, 1536]

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
```

**Parameters:**

config ([MCTCTConfig](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MCTCTConfig](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## MCTCTForCTC[[transformers.MCTCTForCTC]]

#### transformers.MCTCTForCTC[[transformers.MCTCTForCTC]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/modeling_mctct.py#L616)

MCTCT Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.MCTCTForCTC.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/mctct/modeling_mctct.py#L636[{"name": "input_features", "val": ": Tensor"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}]- **input_features** (`torch.LongTensor` of shape `({0})`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [Wav2Vec2CTCTokenizer](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `({0})`, *optional*) --
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

- **labels** (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*) --
  Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
  the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
  All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
  config.vocab_size - 1]`.0[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MCTCTConfig](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MCTCTForCTC](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoProcessor, MCTCTForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("speechbrain/m-ctc-t-large")
>>> model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
"Mr. Quilter is the apostle of the middle classes, and we're glad to welcome his gospel."

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
1885.65
```

**Parameters:**

config ([MCTCTConfig](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MCTCTConfig](/docs/transformers/main/en/model_doc/mctct#transformers.MCTCTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
