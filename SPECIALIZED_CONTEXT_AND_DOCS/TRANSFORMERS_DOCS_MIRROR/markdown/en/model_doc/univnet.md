*This model was released on 2021-06-15 and added to Hugging Face Transformers on 2023-11-22.*

# UnivNet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The UnivNet model was proposed in [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://huggingface.co/papers/2106.07889) by Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kin, and Juntae Kim.
The UnivNet model is a generative adversarial network (GAN) trained to synthesize high fidelity speech waveforms. The UnivNet model shared in `transformers` is the *generator*, which maps a conditioning log-mel spectrogram and optional noise sequence to a speech waveform (e.g. a vocoder). Only the generator is required for inference. The *discriminator* used to train the `generator` is not implemented.

The abstract from the paper is the following:

*Most neural vocoders employ band-limited mel-spectrograms to generate waveforms. If full-band spectral features are used as the input, the vocoder can be provided with as much acoustic information as possible. However, in some models employing full-band mel-spectrograms, an over-smoothing problem occurs as part of which non-sharp spectrograms are generated. To address this problem, we propose UnivNet, a neural vocoder that synthesizes high-fidelity waveforms in real time. Inspired by works in the field of voice activity detection, we added a multi-resolution spectrogram discriminator that employs multiple linear spectrogram magnitudes computed using various parameter sets. Using full-band mel-spectrograms as input, we expect to generate high-resolution signals by adding a discriminator that employs spectrograms of multiple resolutions as the input. In an evaluation on a dataset containing information on hundreds of speakers, UnivNet obtained the best objective and subjective results among competing models for both seen and unseen speakers. These results, including the best subjective score for text-to-speech, demonstrate the potential for fast adaptation to new speakers without a need for training from scratch.*

Tips:

* The `noise_sequence` argument for [UnivNetModel.forward()](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel.forward) should be standard Gaussian noise (such as from `torch.randn`) of shape `([batch_size], noise_length, model.config.model_in_channels)`, where `noise_length` should match the length dimension (dimension 1) of the `input_features` argument. If not supplied, it will be randomly generated; a `torch.Generator` can be supplied to the `generator` argument so that the forward pass can be reproduced. (Note that [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) will return generated noise by default, so it shouldn’t be necessary to generate `noise_sequence` manually.)
* Padding added by [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) can be removed from the [UnivNetModel](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel) output through the `UnivNetFeatureExtractor.batch_decode()` method, as shown in the usage example below.
* Padding the end of each waveform with silence can reduce artifacts at the end of the generated audio sample. This can be done by supplying `pad_end = True` to [UnivNetFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor.__call__). See [this issue](https://github.com/seungwonpark/melgan/issues/8) for more details.

Usage Example:


```
import torch
from scipy.io.wavfile import write
from datasets import Audio, load_dataset

from transformers import UnivNetFeatureExtractor, UnivNetModel

model_id_or_path = "dg845/univnet-dev"
model = UnivNetModel.from_pretrained(model_id_or_path)
feature_extractor = UnivNetFeatureExtractor.from_pretrained(model_id_or_path)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# Resample the audio to the model and feature extractor's sampling rate.
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
# Pad the end of the converted waveforms to reduce artifacts at the end of the output audio samples.
inputs = feature_extractor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], pad_end=True, return_tensors="pt"
)

with torch.no_grad():
    audio = model(**inputs)

# Remove the extra padding at the end of the output.
audio = feature_extractor.batch_decode(**audio)[0]
# Convert to wav file
write("sample_audio.wav", feature_extractor.sampling_rate, audio)
```

This model was contributed by [dg845](https://huggingface.co/dg845).
To the best of my knowledge, there is no official code release, but an unofficial implementation can be found at [maum-ai/univnet](https://github.com/maum-ai/univnet) with pretrained checkpoints [here](https://github.com/maum-ai/univnet#pre-trained-model).

## UnivNetConfig

### class transformers.UnivNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/univnet/configuration_univnet.py#L23)

( model\_in\_channels = 64 model\_hidden\_channels = 32 num\_mel\_bins = 100 resblock\_kernel\_sizes = [3, 3, 3] resblock\_stride\_sizes = [8, 8, 4] resblock\_dilation\_sizes = [[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]] kernel\_predictor\_num\_blocks = 3 kernel\_predictor\_hidden\_channels = 64 kernel\_predictor\_conv\_size = 3 kernel\_predictor\_dropout = 0.0 initializer\_range = 0.01 leaky\_relu\_slope = 0.2 \*\*kwargs  )

Parameters

* **model\_in\_channels** (`int`, *optional*, defaults to 64) —
  The number of input channels for the UnivNet residual network. This should correspond to
  `noise_sequence.shape[1]` and the value used in the [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) class.
* **model\_hidden\_channels** (`int`, *optional*, defaults to 32) —
  The number of hidden channels of each residual block in the UnivNet residual network.
* **num\_mel\_bins** (`int`, *optional*, defaults to 100) —
  The number of frequency bins in the conditioning log-mel spectrogram. This should correspond to the value
  used in the [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) class.
* **resblock\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 3, 3]`) —
  A tuple of integers defining the kernel sizes of the 1D convolutional layers in the UnivNet residual
  network. The length of `resblock_kernel_sizes` defines the number of resnet blocks and should match that of
  `resblock_stride_sizes` and `resblock_dilation_sizes`.
* **resblock\_stride\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 4]`) —
  A tuple of integers defining the stride sizes of the 1D convolutional layers in the UnivNet residual
  network. The length of `resblock_stride_sizes` should match that of `resblock_kernel_sizes` and
  `resblock_dilation_sizes`.
* **resblock\_dilation\_sizes** (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]]`) —
  A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
  UnivNet residual network. The length of `resblock_dilation_sizes` should match that of
  `resblock_kernel_sizes` and `resblock_stride_sizes`. The length of each nested list in
  `resblock_dilation_sizes` defines the number of convolutional layers per resnet block.
* **kernel\_predictor\_num\_blocks** (`int`, *optional*, defaults to 3) —
  The number of residual blocks in the kernel predictor network, which calculates the kernel and bias for
  each location variable convolution layer in the UnivNet residual network.
* **kernel\_predictor\_hidden\_channels** (`int`, *optional*, defaults to 64) —
  The number of hidden channels for each residual block in the kernel predictor network.
* **kernel\_predictor\_conv\_size** (`int`, *optional*, defaults to 3) —
  The kernel size of each 1D convolutional layer in the kernel predictor network.
* **kernel\_predictor\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for each residual block in the kernel predictor network.
* **initializer\_range** (`float`, *optional*, defaults to 0.01) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **leaky\_relu\_slope** (`float`, *optional*, defaults to 0.2) —
  The angle of the negative slope used by the leaky ReLU activation.

This is the configuration class to store the configuration of a [UnivNetModel](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel). It is used to instantiate a
UnivNet vocoder model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the UnivNet
[dg845/univnet-dev](https://huggingface.co/dg845/univnet-dev) architecture, which corresponds to the ‘c32’
architecture in [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/master/config/default_c32.yaml).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import UnivNetModel, UnivNetConfig

>>> # Initializing a Tortoise TTS style configuration
>>> configuration = UnivNetConfig()

>>> # Initializing a model (with random weights) from the Tortoise TTS style configuration
>>> model = UnivNetModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## UnivNetFeatureExtractor

### class transformers.UnivNetFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/univnet/feature_extraction_univnet.py#L29)

( feature\_size: int = 1 sampling\_rate: int = 24000 padding\_value: float = 0.0 do\_normalize: bool = False num\_mel\_bins: int = 100 hop\_length: int = 256 win\_length: int = 1024 win\_function: str = 'hann\_window' filter\_length: typing.Optional[int] = 1024 max\_length\_s: int = 10 fmin: float = 0.0 fmax: typing.Optional[float] = None mel\_floor: float = 1e-09 center: bool = False compression\_factor: float = 1.0 compression\_clip\_val: float = 1e-05 normalize\_min: float = -11.512925148010254 normalize\_max: float = 2.3143386840820312 model\_in\_channels: int = 64 pad\_end\_length: int = 10 return\_attention\_mask = True \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 1) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 24000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value to pad with when applying the padding strategy defined by the `padding` argument to
  [UnivNetFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor.__call__). Should correspond to audio silence. The `pad_end` argument to
  `__call__` will also use this padding value.
* **do\_normalize** (`bool`, *optional*, defaults to `False`) —
  Whether to perform Tacotron 2 normalization on the input. Normalizing can help to significantly improve the
  performance for some models.
* **num\_mel\_bins** (`int`, *optional*, defaults to 100) —
  The number of mel-frequency bins in the extracted spectrogram features. This should match
  `UnivNetModel.config.num_mel_bins`.
* **hop\_length** (`int`, *optional*, defaults to 256) —
  The direct number of samples between sliding windows. Otherwise referred to as “shift” in many papers. Note
  that this is different from other audio feature extractors such as [SpeechT5FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor) which take
  the `hop_length` in ms.
* **win\_length** (`int`, *optional*, defaults to 1024) —
  The direct number of samples for each sliding window. Note that this is different from other audio feature
  extractors such as [SpeechT5FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor) which take the `win_length` in ms.
* **win\_function** (`str`, *optional*, defaults to `"hann_window"`) —
  Name for the window function used for windowing, must be accessible via `torch.{win_function}`
* **filter\_length** (`int`, *optional*, defaults to 1024) —
  The number of FFT components to use. If `None`, this is determined using
  `transformers.audio_utils.optimal_fft_length`.
* **max\_length\_s** (`int`, *optional*, defaults to 10) —
  The maximum input length of the model in seconds. This is used to pad the audio.
* **fmin** (`float`, *optional*, defaults to 0.0) —
  Minimum mel frequency in Hz.
* **fmax** (`float`, *optional*) —
  Maximum mel frequency in Hz. If not set, defaults to `sampling_rate / 2`.
* **mel\_floor** (`float`, *optional*, defaults to 1e-09) —
  Minimum value of mel frequency banks. Note that the way [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) uses `mel_floor` is
  different than in [transformers.audio\_utils.spectrogram()](/docs/transformers/v4.56.2/en/internal/audio_utils#transformers.audio_utils.spectrogram).
* **center** (`bool`, *optional*, defaults to `False`) —
  Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
  `t` will start at time `t * hop_length`.
* **compression\_factor** (`float`, *optional*, defaults to 1.0) —
  The multiplicative compression factor for dynamic range compression during spectral normalization.
* **compression\_clip\_val** (`float`, *optional*, defaults to 1e-05) —
  The clip value applied to the waveform before applying dynamic range compression during spectral
  normalization.
* **normalize\_min** (`float`, *optional*, defaults to -11.512925148010254) —
  The min value used for Tacotron 2-style linear normalization. The default is the original value from the
  Tacotron 2 implementation.
* **normalize\_max** (`float`, *optional*, defaults to 2.3143386840820312) —
  The max value used for Tacotron 2-style linear normalization. The default is the original value from the
  Tacotron 2 implementation.
* **model\_in\_channels** (`int`, *optional*, defaults to 64) —
  The number of input channels to the [UnivNetModel](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel) model. This should match
  `UnivNetModel.config.model_in_channels`.
* **pad\_end\_length** (`int`, *optional*, defaults to 10) —
  If padding the end of each waveform, the number of spectrogram frames worth of samples to append. The
  number of appended samples will be `pad_end_length * hop_length`.
* **return\_attention\_mask** (`bool`, *optional*, defaults to `True`) —
  Whether or not [**call**()](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor.__call__) should return `attention_mask`.

Constructs a UnivNet feature extractor.

This class extracts log-mel-filter bank features from raw speech using the short time Fourier Transform (STFT). The
STFT implementation follows that of TacoTron 2 and Hifi-GAN.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/univnet/feature_extraction_univnet.py#L286)

( raw\_speech: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] sampling\_rate: typing.Optional[int] = None padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True max\_length: typing.Optional[int] = None truncation: bool = True pad\_to\_multiple\_of: typing.Optional[int] = None return\_noise: bool = True generator: typing.Optional[numpy.random.\_generator.Generator] = None pad\_end: bool = False pad\_length: typing.Optional[int] = None do\_normalize: typing.Optional[str] = None return\_attention\_mask: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None  )

Parameters

* **raw\_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
  pipeline.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the input `raw_speech` waveforms (according to the model’s padding side and
  padding index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).

  If `pad_end = True`, that padding will occur before the `padding` strategy is applied.
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **truncation** (`bool`, *optional*, defaults to `True`) —
  Activates truncation to cut input sequences longer than `max_length` to `max_length`.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
* **return\_noise** (`bool`, *optional*, defaults to `True`) —
  Whether to generate and return a noise waveform for use in [UnivNetModel.forward()](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel.forward).
* **generator** (`numpy.random.Generator`, *optional*, defaults to `None`) —
  An optional `numpy.random.Generator` random number generator to use when generating noise.
* **pad\_end** (`bool`, *optional*, defaults to `False`) —
  Whether to pad the end of each waveform with silence. This can help reduce artifacts at the end of the
  generated audio sample; see <https://github.com/seungwonpark/melgan/issues/8> for more details. This
  padding will be done before the padding strategy specified in `padding` is performed.
* **pad\_length** (`int`, *optional*, defaults to `None`) —
  If padding the end of each waveform, the length of the padding in spectrogram frames. If not set, this
  will default to `self.config.pad_end_length`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to perform Tacotron 2 normalization on the input. Normalizing can help to significantly improve
  the performance for some models. If not set, this will default to `self.config.do_normalize`.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature\_extractor’s default.

  [What are attention masks?](../glossary#attention-mask)
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.np.array` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.

Main method to featurize and prepare for the model one or several sequence(s).

## UnivNetModel

### class transformers.UnivNetModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/univnet/modeling_univnet.py#L428)

( config: UnivNetConfig  )

Parameters

* **config** ([UnivNetConfig](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Univnet Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/univnet/modeling_univnet.py#L471)

( input\_features: FloatTensor noise\_sequence: typing.Optional[torch.FloatTensor] = None padding\_mask: typing.Optional[torch.FloatTensor] = None generator: typing.Optional[torch.\_C.Generator] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.univnet.modeling_univnet.UnivNetModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor). See [UnivNetFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor.__call__) for details (`processor_class` uses
  [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) for processing audios).
* **noise\_sequence** (`torch.FloatTensor`, *optional*) —
  Tensor containing a noise sequence of standard Gaussian noise. Can be batched and of shape `(batch_size, sequence_length, config.model_in_channels)`, or un-batched and of shape (sequence\_length,
  config.model\_in\_channels)`. If not supplied, will be randomly generated.
* **padding\_mask** (`torch.BoolTensor`, *optional*) —
  Mask indicating which parts of each sequence are padded. Mask values are selected in `[0, 1]`:
  + 1 for tokens that are **not masked**
  + 0 for tokens that are **masked**

  The mask can be batched and of shape `(batch_size, sequence_length)` or un-batched and of shape
  `(sequence_length,)`.
* **generator** (`torch.Generator`, *optional*) —
  A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
  deterministic.
  return\_dict:
  Whether to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) subclass instead of a plain tuple.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.univnet.modeling_univnet.UnivNetModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.univnet.modeling_univnet.UnivNetModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UnivNetConfig](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetConfig)) and inputs.

* **waveforms** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Batched 1D (mono-channel) output audio waveforms.
* **waveform\_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) — The batched length in samples of each unpadded waveform in `waveforms`.

The [UnivNetModel](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import UnivNetFeatureExtractor, UnivNetModel
>>> from datasets import load_dataset, Audio

>>> model = UnivNetModel.from_pretrained("dg845/univnet-dev")
>>> feature_extractor = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> # Resample the audio to the feature extractor's sampling rate.
>>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> inputs = feature_extractor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> audio = model(**inputs).waveforms
>>> list(audio.shape)
[1, 140288]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/univnet.md)
