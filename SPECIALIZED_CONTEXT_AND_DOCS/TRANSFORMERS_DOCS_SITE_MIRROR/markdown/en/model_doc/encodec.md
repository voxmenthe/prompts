# EnCodec

## Overview

The EnCodec neural codec model was proposed in [High Fidelity Neural Audio Compression](https://huggingface.co/papers/2210.13438) by Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi.

The abstract from the paper is the following:

*We introduce a state-of-the-art real-time, high-fidelity, audio codec leveraging neural networks. It consists in a streaming encoder-decoder architecture with quantized latent space trained in an end-to-end fashion. We simplify and speed-up the training by using a single multiscale spectrogram adversary that efficiently reduces artifacts and produce high-quality samples. We introduce a novel loss balancer mechanism to stabilize training: the weight of a loss now defines the fraction of the overall gradient it should represent, thus decoupling the choice of this hyper-parameter from the typical scale of the loss. Finally, we study how lightweight Transformer models can be used to further compress the obtained representation by up to 40%, while staying faster than real time. We provide a detailed description of the key design choices of the proposed model including: training objective, architectural changes and a study of various perceptual loss functions. We present an extensive subjective evaluation (MUSHRA tests) together with an ablation study for a range of bandwidths and audio domains, including speech, noisy-reverberant speech, and music. Our approach is superior to the baselines methods across all evaluated settings, considering both 24 kHz monophonic and 48 kHz stereophonic audio.*

This model was contributed by [Matthijs](https://huggingface.co/Matthijs), [Patrick Von Platen](https://huggingface.co/patrickvonplaten) and [Arthur Zucker](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/facebookresearch/encodec).

## Usage example

Here is a quick example of how to encode and decode an audio using this model:

```python
>>> from datasets import load_dataset, Audio
>>> from transformers import EncodecModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = EncodecModel.from_pretrained("facebook/encodec_24khz")
>>> processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> # `encoder_outputs.audio_codes` contains discrete codes
>>> audio_values = model.decode(**encoder_outputs, padding_mask=inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

## EncodecConfig[[transformers.EncodecConfig]]

#### transformers.EncodecConfig[[transformers.EncodecConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/configuration_encodec.py#L29)

This is the configuration class to store the configuration of an [EncodecModel](/docs/transformers/main/en/model_doc/encodec#transformers.EncodecModel). It is used to instantiate a
Encodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
[facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import EncodecModel, EncodecConfig

>>> # Initializing a "facebook/encodec_24khz" style configuration
>>> configuration = EncodecConfig()

>>> # Initializing a model (with random weights) from the "facebook/encodec_24khz" style configuration
>>> model = EncodecModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

target_bandwidths (`list[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0, 24.0]`) : The range of different bandwidths the model can encode audio with.

sampling_rate (`int`, *optional*, defaults to 24000) : The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).

audio_channels (`int`, *optional*, defaults to 1) : Number of channels in the audio data. Either 1 for mono or 2 for stereo.

normalize (`bool`, *optional*, defaults to `False`) : Whether the audio shall be normalized when passed.

chunk_length_s (`float`, *optional*) : If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.

overlap (`float`, *optional*) : Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following formulae : `int((1.0 - self.overlap) * self.chunk_length)`.

hidden_size (`int`, *optional*, defaults to 128) : Intermediate representation dimension.

num_filters (`int`, *optional*, defaults to 32) : Number of convolution kernels of first `EncodecConv1d` down sampling layer.

num_residual_layers (`int`,  *optional*, defaults to 1) : Number of residual layers.

upsampling_ratios (`Sequence[int]` , *optional*, defaults to `[8, 5, 4, 2]`) : Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here that must match the decoder order.

norm_type (`str`, *optional*, defaults to `"weight_norm"`) : Normalization method. Should be in `["weight_norm", "time_group_norm"]`

kernel_size (`int`, *optional*, defaults to 7) : Kernel size for the initial convolution.

last_kernel_size (`int`, *optional*, defaults to 7) : Kernel size for the last convolution layer.

residual_kernel_size (`int`, *optional*, defaults to 3) : Kernel size for the residual layers.

dilation_growth_rate (`int`, *optional*, defaults to 2) : How much to increase the dilation with each layer.

use_causal_conv (`bool`, *optional*, defaults to `True`) : Whether to use fully causal convolution.

pad_mode (`str`, *optional*, defaults to `"reflect"`) : Padding mode for the convolutions.

compress (`int`, *optional*, defaults to 2) : Reduced dimensionality in residual branches (from Demucs v3).

num_lstm_layers (`int`, *optional*, defaults to 2) : Number of LSTM layers at the end of the encoder.

trim_right_ratio (`float`, *optional*, defaults to 1.0) : Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If equal to 1.0, it means that all the trimming is done at the right.

codebook_size (`int`, *optional*, defaults to 1024) : Number of discret codes that make up VQVAE.

codebook_dim (`int`, *optional*) : Dimension of the codebook vectors. If not defined, uses `hidden_size`.

use_conv_shortcut (`bool`, *optional*, defaults to `True`) : Whether to use a convolutional layer as the 'skip' connection in the `EncodecResnetBlock` block. If False, an identity function will be used, giving a generic residual connection.

## EncodecFeatureExtractor[[transformers.EncodecFeatureExtractor]]

#### transformers.EncodecFeatureExtractor[[transformers.EncodecFeatureExtractor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/feature_extraction_encodec.py#L29)

Constructs an EnCodec feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
[facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

__call__transformers.EncodecFeatureExtractor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/feature_extraction_encodec.py#L84[{"name": "raw_audio", "val": ": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]]"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy, NoneType] = None"}, {"name": "truncation", "val": ": typing.Optional[bool] = False"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "sampling_rate", "val": ": typing.Optional[int] = None"}]- **raw_audio** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) --
  The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
  `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
  (`feature_size = 2`).
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) --
  Select a strategy to pad the returned sequences (according to the model's padding side and padding
  index) among:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, *optional*, defaults to `False`) --
  Activates truncation to cut input sequences longer than `max_length` to `max_length`.
- **max_length** (`int`, *optional*) --
  Maximum length of the returned list and optionally padding length (see above).
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.
- **sampling_rate** (`int`, *optional*) --
  The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.0

Main method to featurize and prepare for the model one or several sequence(s).

**Parameters:**

feature_size (`int`, *optional*, defaults to 1) : The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.

sampling_rate (`int`, *optional*, defaults to 24000) : The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).

padding_value (`float`, *optional*, defaults to 0.0) : The value that is used to fill the padding values.

chunk_length_s (`float`, *optional*) : If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.

overlap (`float`, *optional*) : Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following formulae : `int((1.0 - self.overlap) * self.chunk_length)`.

## EncodecModel[[transformers.EncodecModel]]

#### transformers.EncodecModel[[transformers.EncodecModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py#L484)

The EnCodec neural audio codec model.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

decodetransformers.EncodecModel.decodehttps://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py#L662[{"name": "audio_codes", "val": ": LongTensor"}, {"name": "audio_scales", "val": ": Tensor"}, {"name": "padding_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "last_frame_pad_length", "val": ": typing.Optional[int] = 0"}]- **audio_codes** (`torch.LongTensor`  of shape `(nb_frames, batch_size, nb_quantizers, frame_len)`, *optional*) --
  Discrete code embeddings computed using `model.encode`.
- **audio_scales** (list of length `nb_frames` of `torch.Tensor` of shape `(batch_size, 1)`, *optional*) --
  Scaling factor for each `audio_codes` input.
- **padding_mask** (`torch.Tensor` of shape `(channels, sequence_length)`) --
  Padding mask used to pad the `input_values`.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **last_frame_pad_length** (`int`, *optional*) --
  Integer representing the length of the padding in the last frame, which is removed during decoding.0

Decodes the given frames into an output audio waveform.

Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
trimmed.

**Parameters:**

config ([EncodecConfig](/docs/transformers/main/en/model_doc/encodec#transformers.EncodecConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
#### encode[[transformers.EncodecModel.encode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py#L526)

Encodes the input audio waveform into discrete codes of shape
`(nb_frames, batch_size, nb_quantizers, frame_len)`.

- `nb_frames=1` if `self.config.chunk_length=None` (as the encoder is applied on the full audio), which is the
case for the 24kHz model. Otherwise, `nb_frames=ceil(input_length/self.config.chunk_stride)`, which is the case
for the 48kHz model.
- `frame_len` is the length of each frame, which is equal to `ceil(input_length/self.config.hop_length)` if
`self.config.chunk_length=None` (e.g., for the 24kHz model). Otherwise, if `self.config.chunk_length` is
defined, `frame_len=self.config.chunk_length/self.config.hop_length`, e.g., the case for the 48kHz model with
`frame_len=150`.

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`) : Float values of the input audio waveform.

padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`) : Padding mask used to pad the `input_values`.

bandwidth (`float`, *optional*) : The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as bandwidth == 6.0

**Returns:**

`EncodecEncoderOutput dict or a tuple containing`

- audio_codes (`torch.LongTensor`  of shape `(nb_frames, batch_size, nb_quantizers, frame_len)`, *optional*),
- audio_scales (list of length `nb_frames` of `torch.Tensor` of shape `(batch_size, 1)`, *optional*),
- last_frame_pad_length (`int`, *optional*).
#### forward[[transformers.EncodecModel.forward]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py#L717)

The [EncodecModel](/docs/transformers/main/en/model_doc/encodec#transformers.EncodecModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from datasets import load_dataset
>>> from transformers import AutoProcessor, EncodecModel

>>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> model_id = "facebook/encodec_24khz"
>>> model = EncodecModel.from_pretrained(model_id)
>>> processor = AutoProcessor.from_pretrained(model_id)

>>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")

>>> outputs = model(**inputs)
>>> audio_codes = outputs.audio_codes
>>> audio_values = outputs.audio_values
```

**Parameters:**

input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*) : Raw audio input converted to Float and padded to the appropriate length in order to be encoded using chunks of length self.chunk_length and a stride of `config.chunk_stride`.

padding_mask (`torch.BoolTensor` of shape `(batch_size, channels, sequence_length)`, *optional*) : Mask to avoid computing scaling factors on padding token indices (can we avoid computing conv on these+). Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.    `padding_mask` should always be passed, unless the input was truncated or not padded. This is because in order to process tensors effectively, the input audio should be padded so that `input_length % stride = step` with `step = chunk_length-stride`. This ensures that all chunks are of the same shape  

bandwidth (`float`, *optional*) : The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as `bandwidth == 6.0`

audio_codes (`torch.LongTensor`  of shape `(nb_frames, batch_size, nb_quantizers, frame_len)`, *optional*) : Discrete code embeddings computed using `model.encode`.

audio_scales (`list` of length `nb_frames` of `torch.Tensor` of shape `(batch_size, 1)`, *optional*) : Scaling factor for each `audio_codes` input.

return_dict (`bool`, *optional*) : Whether to return outputs as a dict.

last_frame_pad_length (`int`, *optional*) : The length of the padding in the last frame, if any. This is used to ensure that the encoded frames can be outputted as a tensor. This value should be passed during decoding to ensure padding is removed from the encoded frames.

**Returns:**

``transformers.models.encodec.modeling_encodec.EncodecOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.encodec.modeling_encodec.EncodecOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EncodecConfig](/docs/transformers/main/en/model_doc/encodec#transformers.EncodecConfig)) and inputs.

- **audio_codes** (`torch.LongTensor`  of shape `(nb_frames, batch_size, nb_quantizers, frame_len)`, *optional*) -- Discrete code embeddings computed using `model.encode`.
- **audio_values** (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*) -- Decoded audio values, obtained using the decoder part of Encodec.
