*This model was released on 2023-06-11 and added to Hugging Face Transformers on 2024-08-19.*

# DAC

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The DAC model was proposed in [Descript Audio Codec: High-Fidelity Audio Compression with Improved RVQGAN](https://huggingface.co/papers/2306.06546) by Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar.

The Descript Audio Codec (DAC) model is a powerful tool for compressing audio data, making it highly efficient for storage and transmission. By compressing 44.1 KHz audio into tokens at just 8kbps bandwidth, the DAC model enables high-quality audio processing while significantly reducing the data footprint. This is particularly useful in scenarios where bandwidth is limited or storage space is at a premium, such as in streaming applications, remote conferencing, and archiving large audio datasets.

The abstract from the paper is the following:

*Language models have been successfully used to model natural signals, such as images, speech, and music. A key component of these models is a high quality neural compression model that can compress high-dimensional natural signals into lower dimensional discrete tokens. To that end, we introduce a high-fidelity universal neural audio compression algorithm that achieves ~90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth. We achieve this by combining advances in high-fidelity audio generation with better vector quantization techniques from the image domain, along with improved adversarial and reconstruction losses. We compress all domains (speech, environment, music, etc.) with a single universal model, making it widely applicable to generative modeling of all audio. We compare with competing audio compression algorithms, and find our method outperforms them significantly. We provide thorough ablations for every design choice, as well as open-source code and trained model weights. We hope our work can lay the foundation for the next generation of high-fidelity audio modeling.*

This model was contributed by [Kamil Akesbi](https://huggingface.co/kamilakesbi).
The original code can be found [here](https://github.com/descriptinc/descript-audio-codec/tree/main?tab=readme-ov-file).

## Model structure

The Descript Audio Codec (DAC) model is structured into three distinct stages:

1. Encoder Model: This stage compresses the input audio, reducing its size while retaining essential information.
2. Residual Vector Quantizer (RVQ) Model: Working in tandem with the encoder, this model quantizes the latent codes of the audio, refining the compression and ensuring high-quality reconstruction.
3. Decoder Model: This final stage reconstructs the audio from its compressed form, restoring it to a state that closely resembles the original input.

## Usage example

Here is a quick example of how to encode and decode an audio using this model:


```
>>> from datasets import load_dataset, Audio
>>> from transformers import DacModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = DacModel.from_pretrained("descript/dac_16khz")
>>> processor = AutoProcessor.from_pretrained("descript/dac_16khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"])
>>> # Get the intermediate audio codes
>>> audio_codes = encoder_outputs.audio_codes
>>> # Reconstruct the audio from its quantized representation
>>> audio_values = model.decode(encoder_outputs.quantized_representation)
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"]).audio_values
```

## DacConfig

### class transformers.DacConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/configuration_dac.py#L28)

( encoder\_hidden\_size = 64 downsampling\_ratios = [2, 4, 8, 8] decoder\_hidden\_size = 1536 n\_codebooks = 9 codebook\_size = 1024 codebook\_dim = 8 quantizer\_dropout = 0 commitment\_loss\_weight = 0.25 codebook\_loss\_weight = 1.0 sampling\_rate = 16000 \*\*kwargs  )

Parameters

* **encoder\_hidden\_size** (`int`, *optional*, defaults to 64) —
  Intermediate representation dimension for the encoder.
* **downsampling\_ratios** (`list[int]`, *optional*, defaults to `[2, 4, 8, 8]`) —
  Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
* **decoder\_hidden\_size** (`int`, *optional*, defaults to 1536) —
  Intermediate representation dimension for the decoder.
* **n\_codebooks** (`int`, *optional*, defaults to 9) —
  Number of codebooks in the VQVAE.
* **codebook\_size** (`int`, *optional*, defaults to 1024) —
  Number of discrete codes in each codebook.
* **codebook\_dim** (`int`, *optional*, defaults to 8) —
  Dimension of the codebook vectors. If not defined, uses `encoder_hidden_size`.
* **quantizer\_dropout** (`bool`, *optional*, defaults to 0) —
  Whether to apply dropout to the quantizer.
* **commitment\_loss\_weight** (float, *optional*, defaults to 0.25) —
  Weight of the commitment loss term in the VQVAE loss function.
* **codebook\_loss\_weight** (float, *optional*, defaults to 1.0) —
  Weight of the codebook loss term in the VQVAE loss function.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).

This is the configuration class to store the configuration of an [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel). It is used to instantiate a
Dac model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
[descript/dac\_16khz](https://huggingface.co/descript/dac_16khz) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DacModel, DacConfig

>>> # Initializing a "descript/dac_16khz" style configuration
>>> configuration = DacConfig()

>>> # Initializing a model (with random weights) from the "descript/dac_16khz" style configuration
>>> model = DacModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DacFeatureExtractor

### class transformers.DacFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/feature_extraction_dac.py#L29)

( feature\_size: int = 1 sampling\_rate: int = 16000 padding\_value: float = 0.0 hop\_length: int = 512 \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 1) —
  The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used for padding.
* **hop\_length** (`int`, *optional*, defaults to 512) —
  Overlap length between successive windows.

Constructs an Dac feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/feature_extraction_dac.py#L60)

( raw\_audio: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy, NoneType] = None truncation: typing.Optional[bool] = False max\_length: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None sampling\_rate: typing.Optional[int] = None  )

Parameters

* **raw\_audio** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
  `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
  (`feature_size = 2`).
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, *optional*, defaults to `False`) —
  Activates truncation to cut input sequences longer than `max_length` to `max_length`.
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*, default to ‘pt’) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.

Main method to featurize and prepare for the model one or several sequence(s).

## DacModel

### class transformers.DacModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/modeling_dac.py#L558)

( config: DacConfig  )

Parameters

* **config** ([DacConfig](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The DAC (Descript Audio Codec) model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/modeling_dac.py#L602)

( quantized\_representation: typing.Optional[torch.Tensor] = None audio\_codes: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **quantized\_representation** (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`, *optional*) —
  Quantized continuous representation of input.
* **audio\_codes** (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`, *optional*) —
  The codebook indices for each codebook, representing the quantized discrete
  representation of the input. This parameter should be provided if you want
  to decode directly from the audio codes (it will overwrite quantized\_representation).
* **return\_dict** (`bool`, *optional*, defaults to `True`) —
  Whether to return a `DacDecoderOutput` instead of a plain tuple.

#### encode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/modeling_dac.py#L575)

( input\_values: Tensor n\_quantizers: typing.Optional[int] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **input\_values** (`torch.Tensor of shape` (batch\_size, 1, time\_steps)`) —
  Input audio data to encode,
* **n\_quantizers** (`int`, *optional*) —
  Number of quantizers to use. If None, all quantizers are used. Default is None.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dac/modeling_dac.py#L635)

( input\_values: Tensor n\_quantizers: typing.Optional[int] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, 1, time_steps)`) —
  Audio data to encode.
* **n\_quantizers** (`int`, *optional*) —
  Number of quantizers to use. If `None`, all quantizers are used. Default is `None`.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from datasets import load_dataset, Audio
>>> from transformers import DacModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = DacModel.from_pretrained("descript/dac_16khz")
>>> processor = AutoProcessor.from_pretrained("descript/dac_16khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"])
>>> # Get the intermediate audio codes
>>> audio_codes = encoder_outputs.audio_codes
>>> # Reconstruct the audio from its quantized representation
>>> audio_values = model.decode(encoder_outputs.quantized_representation)
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"]).audio_values
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dac.md)
