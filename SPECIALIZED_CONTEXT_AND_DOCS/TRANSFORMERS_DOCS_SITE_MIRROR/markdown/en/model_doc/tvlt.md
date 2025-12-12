# TVLT

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The TVLT model was proposed in [TVLT: Textless Vision-Language Transformer](https://huggingface.co/papers/2209.14156)
by Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal (the first three authors contributed equally). The Textless Vision-Language Transformer (TVLT) is a model that uses raw visual and audio inputs for vision-and-language representation learning, without using text-specific modules such as tokenization or automatic speech recognition (ASR). It can perform various audiovisual and vision-language tasks like retrieval, question answering, etc.

The abstract from the paper is the following:

*In this work, we present the Textless Vision-Language Transformer (TVLT), where homogeneous transformer blocks take raw visual and audio inputs for vision-and-language representation learning with minimal modality-specific design, and do not use text-specific modules such as tokenization or automatic speech recognition (ASR). TVLT is trained by reconstructing masked patches of continuous video frames and audio spectrograms (masked autoencoding) and contrastive modeling to align video and audio. TVLT attains performance comparable to its text-based counterpart on various multimodal tasks, such as visual question answering, image retrieval, video retrieval, and multimodal sentiment analysis, with 28x faster inference speed and only 1/3 of the parameters. Our findings suggest the possibility of learning compact and efficient visual-linguistic representations from low-level visual and audio signals without assuming the prior existence of text.*

 TVLT architecture. Taken from the original paper. 

The original code can be found [here](https://github.com/zinengtang/TVLT). This model was contributed by [Zineng Tang](https://huggingface.co/ZinengTang).

## Usage tips

- TVLT is a model that takes both `pixel_values` and `audio_values` as input. One can use [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor) to prepare data for the model.
  This processor wraps an image processor (for the image/video modality) and an audio feature extractor (for the audio modality) into one.
- TVLT is trained with images/videos and audios of various sizes: the authors resize and crop the input images/videos to 224 and limit the length of audio spectrogram to 2048. To make batching of videos and audios possible, the authors use a `pixel_mask` that indicates which pixels are real/padding and `audio_mask` that indicates which audio values are real/padding.
- The design of TVLT is very similar to that of a standard Vision Transformer (ViT) and masked autoencoder (MAE) as in [ViTMAE](vitmae). The difference is that the model includes embedding layers for the audio modality.
- The PyTorch version of this model is only available in torch 1.10 and higher.

## TvltConfig[[transformers.TvltConfig]]

#### transformers.TvltConfig[[transformers.TvltConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/configuration_tvlt.py#L24)

This is the configuration class to store the configuration of a [TvltModel](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltModel). It is used to instantiate a TVLT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the TVLT
[ZinengTang/tvlt-base](https://huggingface.co/ZinengTang/tvlt-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import TvltConfig, TvltModel

>>> # # Initializing a TVLT ZinengTang/tvlt-base style configuration
>>> configuration = TvltConfig()

>>> # # Initializing a model (with random weights) from the ZinengTang/tvlt-base style configuration
>>> model = TvltModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

spectrogram_length (`int`, *optional*, defaults to 2048) : The time length of each audio spectrogram.

frequency_length (`int`, *optional*, defaults to 128) : The frequency length of audio spectrogram.

image_patch_size (`list[int]`, *optional*, defaults to `[16, 16]`) : The size (resolution) of each image patch.

audio_patch_size (`list[int]`, *optional*, defaults to `[16, 16]`) : The size (resolution) of each audio patch.

num_image_channels (`int`, *optional*, defaults to 3) : The number of input image channels.

num_audio_channels (`int`, *optional*, defaults to 1) : The number of input audio channels.

num_frames (`int`, *optional*, defaults to 8) : The maximum number of frames for an input video.

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

use_mean_pooling (`bool`, *optional*, defaults to `False`) : Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.

decoder_num_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the decoder.

decoder_hidden_size (`int`, *optional*, defaults to 512) : Dimensionality of the decoder.

decoder_num_hidden_layers (`int`, *optional*, defaults to 8) : Number of hidden layers in the decoder.

decoder_intermediate_size (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.

pixel_mask_ratio (`float`, *optional*, defaults to 0.75) : Image patch masking ratio.

audio_mask_ratio (`float`, *optional*, defaults to 0.15) : Audio patch masking ratio.

audio_mask_type (`str`, *optional*, defaults to `"frame-level"`) : Audio patch masking type, choose between "frame-level" and "patch-level".

task_matching (`bool`, *optional*, defaults to `True`) : Whether to use vision audio matching task in pretraining.

task_mae (`bool`, *optional*, defaults to `True`) : Whether to use the masked auto-encoder (MAE) in pretraining.

loss_type (`str`, *optional*, defaults to `"classification"`) : Loss types including regression and classification.

## TvltProcessor[[transformers.TvltProcessor]]

#### transformers.TvltProcessor[[transformers.TvltProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/processing_tvlt.py#L22)

Constructs a TVLT processor which wraps a TVLT image processor and TVLT feature extractor into a single processor.

[TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor) offers all the functionalities of [TvltImageProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltImageProcessor) and [TvltFeatureExtractor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltFeatureExtractor). See the
docstring of [__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for more information.

__call__transformers.TvltProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/processing_tvlt.py#L46[{"name": "images", "val": " = None"}, {"name": "audio", "val": " = None"}, {"name": "images_mixed", "val": " = None"}, {"name": "sampling_rate", "val": " = None"}, {"name": "mask_audio", "val": " = False"}, {"name": "mask_pixel", "val": " = False"}, {"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]

Forwards the `images` argument to TvltImageProcessor's [preprocess()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltImageProcessor.preprocess) and the `audio`
argument to TvltFeatureExtractor's [__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltFeatureExtractor.__call__). Please refer to the docstring of the
above two methods for more information.

**Parameters:**

image_processor (`TvltImageProcessor`) : An instance of [TvltImageProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltImageProcessor). The image processor is a required input.

feature_extractor (`TvltFeatureExtractor`) : An instance of [TvltFeatureExtractor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltFeatureExtractor). The feature extractor is a required input.

## TvltFeatureExtractor[[transformers.TvltFeatureExtractor]]

#### transformers.TvltFeatureExtractor[[transformers.TvltFeatureExtractor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/feature_extraction_tvlt.py#L30)

Constructs a TVLT audio feature extractor. This feature extractor can be used to prepare audios for the model.

This feature extractor inherits from [FeatureExtractionMixin](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin) which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.

__call__transformers.TvltFeatureExtractor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/feature_extraction_tvlt.py#L116[{"name": "raw_speech", "val": ": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]]"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = True"}, {"name": "sampling_rate", "val": ": typing.Optional[int] = None"}, {"name": "resample", "val": ": bool = False"}, {"name": "mask_audio", "val": ": bool = False"}, {"name": "**kwargs", "val": ""}]- **raw_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) --
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:
  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.
- **return_attention_mask** (`bool`, *optional*, default to `True`) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature_extractor's default. [What are attention masks?](../glossary#attention-mask)

  

  For TvltTransformer models, `attention_mask` should always be passed for batched inference, to avoid
  subtle bugs.

  

- **sampling_rate** (`int`, *optional*) --
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
  pipeline. Current model supports sampling rate 16000 and 44100.
- **resample** (`bool`, *optional*, defaults to `False`) --
  If the sampling rate is not matched, resample the input audio to match.
- **mask_audio** (`bool`, *optional*, defaults to `False`) --
  Whether or not to mask input audio for MAE task.0[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

- **audio_values** -- Audio values to be fed to a model, of shape (batch_size, num_channels, height,
  width).

- **audio_mask** -- Audio masks to be fed to a model, of shape (batch_size, num_audio_patches).

Main method to prepare one or several audio(s) for the model.

**Parameters:**

spectrogram_length (`dict[str, int]` *optional*, defaults to 2048) : The time length of each audio spectrogram.

num_channels (`int` *optional*, defaults to 1) : Number of audio channels.

patch_size (`list[int]` *optional*, defaults to `[16, 16]`) : The patch size of audio patch embedding.

feature_size (`int`, *optional*, defaults to 128) : The frequency length of audio spectrogram.

sampling_rate (`int`, *optional*, defaults to 44100) : The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz).

hop_length_to_sampling_rate (`int`, *optional*, defaults to 86) : Hop length is length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients. For example, with sampling rate 44100, the hop length is 512, with 44100 / 512 = 86

n_fft (`int`, *optional*, defaults to 2048) : Size of the Fourier transform.

padding_value (`float`, *optional*, defaults to 0.0) : Padding value used to pad the audio. Should correspond to silences.

**Returns:**

`[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)`

A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

- **audio_values** -- Audio values to be fed to a model, of shape (batch_size, num_channels, height,
  width).

- **audio_mask** -- Audio masks to be fed to a model, of shape (batch_size, num_audio_patches).

## TvltImageProcessor[[transformers.TvltImageProcessor]]

#### transformers.TvltImageProcessor[[transformers.TvltImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/image_processing_tvlt.py#L70)

Constructs a TVLT image processor.

This processor can be used to prepare either videos or images for the model by converting images to 1-frame videos.

preprocesstransformers.TvltImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/image_processing_tvlt.py#L277[{"name": "videos", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "patch_size", "val": ": typing.Optional[list[int]] = None"}, {"name": "num_frames", "val": ": typing.Optional[int] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "is_mixed", "val": ": bool = False"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}, {"name": "**kwargs", "val": ""}]- **videos** (`ImageInput`) --
  Images or videos to preprocess. Expects a single or batch of frames with pixel values ranging from 0 to
  255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image after applying resize.
- **patch_size** (`list[int]` *optional*, defaults to self.patch_size) --
  The patch size of image patch embedding.
- **num_frames** (`int` *optional*, defaults to self.num_frames) --
  The maximum number of video frames.
- **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_centre_crop`) --
  Whether to centre crop the image.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  Size of the image after applying the centre crop.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image values between [0 - 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation.
- **is_mixed** (`bool`, *optional*) --
  If the input video has negative samples.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - Unset: Use the inferred channel dimension format of the input image.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

- **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
  width).

- **pixel_mask** -- Pixel masks to be fed to a model, of shape (batch_size, num_pixel_patches).

- **pixel_values_mixed** -- Pixel values with both positive or negative to be fed to a model, of shape
  (batch_size, num_channels, height, width).

- **pixel_mask_mixed** -- Pixel masks with both positive or negative to be fed to a model, of shape
  (batch_size, num_pixel_patches).

Preprocess an videos or image or batch of videos or images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in the `preprocess` method.

size (`dict[str, int]` *optional*, defaults to `{"shortest_edge" : 224}`): Size of the output image after resizing. The shortest edge of the image will be resized to `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overridden by `size` in the `preprocess` method.

patch_size (`list[int]` *optional*, defaults to [16,16]) : The patch size of image patch embedding.

num_frames (`int` *optional*, defaults to 8) : The maximum number of video frames.

resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`) : Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.

do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop` parameter in the `preprocess` method.

crop_size (`dict[str, int]`, *optional*, defaults to `{"height" : 224, "width": 224}`): Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to 1/255) : Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

**Returns:**

`[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)`

A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

- **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
  width).

- **pixel_mask** -- Pixel masks to be fed to a model, of shape (batch_size, num_pixel_patches).

- **pixel_values_mixed** -- Pixel values with both positive or negative to be fed to a model, of shape
  (batch_size, num_channels, height, width).

- **pixel_mask_mixed** -- Pixel masks with both positive or negative to be fed to a model, of shape
  (batch_size, num_pixel_patches).

## TvltModel[[transformers.TvltModel]]

#### transformers.TvltModel[[transformers.TvltModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/modeling_tvlt.py#L612)

The bare TVLT Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.TvltModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/modeling_tvlt.py#L642[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "audio_values", "val": ": FloatTensor"}, {"name": "pixel_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "audio_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "mask_pixel", "val": ": bool = False"}, {"name": "mask_audio", "val": ": bool = False"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **audio_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Audio values. Audio values can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **pixel_mask** (`torch.FloatTensor` of shape `(batch_size, num_pixel_patches)`) --
  Pixel masks. Pixel masks can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **audio_mask** (`torch.FloatTensor` of shape `(batch_size, num_audio_patches)`) --
  Audio masks. Audio masks can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **pixel_values_mixed** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Pixel values mixed can
  be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **pixel_mask_mixed** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel masks of pixel_values_mixed. Pixel masks mixed can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See
  [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **mask_pixel** (`bool`, *optional*) --
  Whether to mask pixel for MAE tasks. Only set to True in TvltForPreTraining.

- **mask_audio** (`bool`, *optional*) --
  Whether to mask audio for MAE tasks. Only set to True in TvltForPreTraining.

- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.

- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.deprecated.tvlt.modeling_tvlt.TvltModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.tvlt.modeling_tvlt.TvltModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **last_pixel_hidden_state** (`torch.FloatTensor` of shape `(batch_size, pixel_sequence_length, hidden_size)`) -- Pixel sequence of hidden-states at the output of the last layer of the model.
- **last_audio_hidden_state** (`torch.FloatTensor` of shape `(batch_size, audio_sequence_length, hidden_size)`) -- Audio sequence of hidden-states at the output of the last layer of the model.
- **pixel_label_masks** (`torch.FloatTensor` of shape `(batch_size, pixel_patch_length)`) -- Tensor indicating which pixel patches are masked (1) and which are not (0).
- **audio_label_masks** (`torch.FloatTensor` of shape `(batch_size, audio_patch_length)`) -- Tensor indicating which audio patches are masked (1) and which are not (0).
- **pixel_ids_restore** (`torch.LongTensor` of shape `(batch_size, pixel_patch_length)`) -- Tensor containing the ids permutation of pixel masking.
- **audio_ids_restore** (`torch.LongTensor` of shape `(batch_size, audio_patch_length)`) -- Tensor containing the ids permutation of audio masking.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
The [TvltModel](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import TvltProcessor, TvltModel
>>> import numpy as np
>>> import torch

>>> num_frames = 8
>>> images = list(np.random.randn(num_frames, 3, 224, 224))
>>> audio = list(np.random.randn(10000))

>>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
>>> model = TvltModel.from_pretrained("ZinengTang/tvlt-base")

>>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

>>> outputs = model(**input_dict)
>>> loss = outputs.loss
```

**Parameters:**

config ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.tvlt.modeling_tvlt.TvltModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.tvlt.modeling_tvlt.TvltModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **last_pixel_hidden_state** (`torch.FloatTensor` of shape `(batch_size, pixel_sequence_length, hidden_size)`) -- Pixel sequence of hidden-states at the output of the last layer of the model.
- **last_audio_hidden_state** (`torch.FloatTensor` of shape `(batch_size, audio_sequence_length, hidden_size)`) -- Audio sequence of hidden-states at the output of the last layer of the model.
- **pixel_label_masks** (`torch.FloatTensor` of shape `(batch_size, pixel_patch_length)`) -- Tensor indicating which pixel patches are masked (1) and which are not (0).
- **audio_label_masks** (`torch.FloatTensor` of shape `(batch_size, audio_patch_length)`) -- Tensor indicating which audio patches are masked (1) and which are not (0).
- **pixel_ids_restore** (`torch.LongTensor` of shape `(batch_size, pixel_patch_length)`) -- Tensor containing the ids permutation of pixel masking.
- **audio_ids_restore** (`torch.LongTensor` of shape `(batch_size, audio_patch_length)`) -- Tensor containing the ids permutation of audio masking.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

## TvltForPreTraining[[transformers.TvltForPreTraining]]

#### transformers.TvltForPreTraining[[transformers.TvltForPreTraining]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/modeling_tvlt.py#L829)

The TVLT Model transformer with the decoder on top for self-supervised pre-training.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.TvltForPreTraining.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/modeling_tvlt.py#L962[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "audio_values", "val": ": FloatTensor"}, {"name": "pixel_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "audio_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values_mixed", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "pixel_mask_mixed", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **audio_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Audio values. Audio values can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **pixel_mask** (`torch.FloatTensor` of shape `(batch_size, num_pixel_patches)`) --
  Pixel masks. Pixel masks can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **audio_mask** (`torch.FloatTensor` of shape `(batch_size, num_audio_patches)`) --
  Audio masks. Audio masks can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **pixel_values_mixed** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Pixel values mixed can
  be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **pixel_mask_mixed** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel masks of pixel_values_mixed. Pixel masks mixed can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See
  [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **mask_pixel** (`bool`, *optional*) --
  Whether to mask pixel for MAE tasks. Only set to True in TvltForPreTraining.

- **mask_audio** (`bool`, *optional*) --
  Whether to mask audio for MAE tasks. Only set to True in TvltForPreTraining.

- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.

- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **pixel_values_mixed** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Audio values can be
  obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **pixel_mask_mixed** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel masks of pixel_values_mixed. Pixel values mixed can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See
  [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **labels** (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*) --
  Labels for computing the vision audio matching loss. Indices should be in `[0, 1]`. num_labels has to be 1.0`transformers.models.deprecated.tvlt.modeling_tvlt.TvltForPreTrainingOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.tvlt.modeling_tvlt.TvltForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`) -- Pixel reconstruction loss.
- **matching_logits** (`torch.FloatTensor` of shape `(batch_size, 1)`) -- Matching objective logits.
- **pixel_logits** (`torch.FloatTensor` of shape
  `(batch_size, pixel_patch_length, image_patch_size ** 3 * pixel_num_channels)`): Pixel reconstruction
  logits.
- **audio_logits** (`torch.FloatTensor` of shape
  `(batch_size, audio_patch_length, image_patch_size[0] * image_patch_size[1])`): Audio reconstruction
  logits.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
The [TvltForPreTraining](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import TvltProcessor, TvltForPreTraining
>>> import numpy as np
>>> import torch

>>> num_frames = 8
>>> images = list(np.random.randn(num_frames, 3, 224, 224))
>>> images_mixed = list(np.random.randn(num_frames, 3, 224, 224))
>>> audio = list(np.random.randn(10000))
>>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
>>> model = TvltForPreTraining.from_pretrained("ZinengTang/tvlt-base")
>>> input_dict = processor(
...     images, audio, images_mixed, sampling_rate=44100, mask_pixel=True, mask_audio=True, return_tensors="pt"
... )

>>> outputs = model(**input_dict)
>>> loss = outputs.loss
```

**Parameters:**

config ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.tvlt.modeling_tvlt.TvltForPreTrainingOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.tvlt.modeling_tvlt.TvltForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`) -- Pixel reconstruction loss.
- **matching_logits** (`torch.FloatTensor` of shape `(batch_size, 1)`) -- Matching objective logits.
- **pixel_logits** (`torch.FloatTensor` of shape
  `(batch_size, pixel_patch_length, image_patch_size ** 3 * pixel_num_channels)`): Pixel reconstruction
  logits.
- **audio_logits** (`torch.FloatTensor` of shape
  `(batch_size, audio_patch_length, image_patch_size[0] * image_patch_size[1])`): Audio reconstruction
  logits.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

## TvltForAudioVisualClassification[[transformers.TvltForAudioVisualClassification]]

#### transformers.TvltForAudioVisualClassification[[transformers.TvltForAudioVisualClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/modeling_tvlt.py#L1145)

Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the [CLS] token)
for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.TvltForAudioVisualClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tvlt/modeling_tvlt.py#L1163[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "audio_values", "val": ": FloatTensor"}, {"name": "pixel_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "audio_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **audio_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Audio values. Audio values can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **pixel_mask** (`torch.FloatTensor` of shape `(batch_size, num_pixel_patches)`) --
  Pixel masks. Pixel masks can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **audio_mask** (`torch.FloatTensor` of shape `(batch_size, num_audio_patches)`) --
  Audio masks. Audio masks can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for
  details.

- **pixel_values_mixed** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`) --
  Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Pixel values mixed can
  be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **pixel_mask_mixed** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel masks of pixel_values_mixed. Pixel masks mixed can be obtained using [TvltProcessor](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor). See
  [TvltProcessor.__call__()](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for details.

- **mask_pixel** (`bool`, *optional*) --
  Whether to mask pixel for MAE tasks. Only set to True in TvltForPreTraining.

- **mask_audio** (`bool`, *optional*) --
  Whether to mask audio for MAE tasks. Only set to True in TvltForPreTraining.

- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.

- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **labels** (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*) --
  Labels for computing the audiovisual loss. Indices should be in `[0, ..., num_classes-1]` where num_classes
  refers to the number of classes in audiovisual tasks.0[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [TvltForAudioVisualClassification](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltForAudioVisualClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
>>> from transformers import TvltProcessor, TvltForAudioVisualClassification
>>> import numpy as np
>>> import torch

>>> num_frames = 8
>>> images = list(np.random.randn(num_frames, 3, 224, 224))
>>> audio = list(np.random.randn(10000))
>>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
>>> model = TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")
>>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

>>> outputs = model(**input_dict)
>>> loss = outputs.loss
```

**Parameters:**

config ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TvltConfig](/docs/transformers/main/en/model_doc/tvlt#transformers.TvltConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
