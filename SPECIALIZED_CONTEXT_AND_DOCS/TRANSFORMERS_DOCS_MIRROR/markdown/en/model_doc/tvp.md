*This model was released on 2023-03-09 and added to Hugging Face Transformers on 2023-11-22.*

# TVP

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The text-visual prompting (TVP) framework was proposed in the paper [Text-Visual Prompting for Efficient 2D Temporal Video Grounding](https://huggingface.co/papers/2303.04995) by Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding.

The abstract from the paper is the following:

*In this paper, we study the problem of temporal video grounding (TVG), which aims to predict the starting/ending time points of moments described by a text sentence within a long untrimmed video. Benefiting from fine-grained 3D visual features, the TVG techniques have achieved remarkable progress in recent years. However, the high complexity of 3D convolutional neural networks (CNNs) makes extracting dense 3D visual features time-consuming, which calls for intensive memory and computing resources. Towards efficient TVG, we propose a novel text-visual prompting (TVP) framework, which incorporates optimized perturbation patterns (that we call ‘prompts’) into both visual inputs and textual features of a TVG model. In sharp contrast to 3D CNNs, we show that TVP allows us to effectively co-train vision encoder and language encoder in a 2D TVG model and improves the performance of cross-modal feature fusion using only low-complexity sparse 2D visual features. Further, we propose a Temporal-Distance IoU (TDIoU) loss for efficient learning of TVG. Experiments on two benchmark datasets, Charades-STA and ActivityNet Captions datasets, empirically show that the proposed TVP significantly boosts the performance of 2D TVG (e.g., 9.79% improvement on Charades-STA and 30.77% improvement on ActivityNet Captions) and achieves 5× inference acceleration over TVG using 3D visual features.*

This research addresses temporal video grounding (TVG), which is the process of pinpointing the start and end times of specific events in a long video, as described by a text sentence. Text-visual prompting (TVP), is proposed to enhance TVG. TVP involves integrating specially designed patterns, known as ‘prompts’, into both the visual (image-based) and textual (word-based) input components of a TVG model. These prompts provide additional spatial-temporal context, improving the model’s ability to accurately determine event timings in the video. The approach employs 2D visual inputs in place of 3D ones. Although 3D inputs offer more spatial-temporal detail, they are also more time-consuming to process. The use of 2D inputs with the prompting method aims to provide similar levels of context and accuracy more efficiently.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvp_architecture.png) TVP architecture. Taken from the [original paper.](https://huggingface.co/papers/2303.04995)

This model was contributed by [Jiqing Feng](https://huggingface.co/Jiqing). The original code can be found [here](https://github.com/intel/TVP).

## Usage tips and examples

Prompts are optimized perturbation patterns, which would be added to input video frames or text features. Universal set refers to using the same exact set of prompts for any input, this means that these prompts are added consistently to all video frames and text features, regardless of the input’s content.

TVP consists of a visual encoder and cross-modal encoder. A universal set of visual prompts and text prompts to be integrated into sampled video frames and textual features, respectively. Specially, a set of different visual prompts are applied to uniformly-sampled frames of one untrimmed video in order.

The goal of this model is to incorporate trainable prompts into both visual inputs and textual features to temporal video grounding(TVG) problems.
In principle, one can apply any visual, cross-modal encoder in the proposed architecture.

The [TvpProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor) wraps [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) and [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor) into a single instance to both
encode the text and prepare the images respectively.

The following example shows how to run temporal video grounding using [TvpProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor) and [TvpForVideoGrounding](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpForVideoGrounding).


```
import av
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, TvpForVideoGrounding


def pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
    '''
    Convert the video from its original fps to the target_fps and decode the video with PyAV decoder.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling.
            If clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the given video.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
    '''
    video = container.streams.video[0]
    fps = float(video.average_rate)
    clip_size = sampling_rate * num_frames / target_fps * fps
    delta = max(num_frames - clip_size, 0)
    start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    timebase = video.duration / num_frames
    video_start_pts = int(start_idx * timebase)
    video_end_pts = int(end_idx * timebase)
    seek_offset = max(video_start_pts - 1024, 0)
    container.seek(seek_offset, any_frame=False, backward=True, stream=video)
    frames = {}
    for frame in container.decode(video=0):
        if frame.pts < video_start_pts:
            continue
        frames[frame.pts] = frame
        if frame.pts > video_end_pts:
            break
    frames = [frames[pts] for pts in sorted(frames)]
    return frames, fps


def decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
    '''
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling.
            If clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the given video.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video.
    '''
    assert clip_idx >= -2, "Not a valid clip_idx {}".format(clip_idx)
    frames, fps = pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps)
    clip_size = sampling_rate * num_frames / target_fps * fps
    index = np.linspace(0, clip_size - 1, num_frames)
    index = np.clip(index, 0, len(frames) - 1).astype(np.int64)
    frames = np.array([frames[idx].to_rgb().to_ndarray() for idx in index])
    frames = frames.transpose(0, 3, 1, 2)
    return frames


file = hf_hub_download(repo_id="Intel/tvp_demo", filename="AK2KG.mp4", repo_type="dataset")
model = TvpForVideoGrounding.from_pretrained("Intel/tvp-base")

decoder_kwargs = dict(
    container=av.open(file, metadata_errors="ignore"),
    sampling_rate=1,
    num_frames=model.config.num_frames,
    clip_idx=0,
    num_clips=1,
    target_fps=3,
)
raw_sampled_frms = decode(**decoder_kwargs)

text = "a person is sitting on a bed."
processor = AutoProcessor.from_pretrained("Intel/tvp-base")
model_inputs = processor(
    text=[text], videos=list(raw_sampled_frms), return_tensors="pt", max_text_length=100#, size=size
)

model_inputs["pixel_values"] = model_inputs["pixel_values"].to(model.dtype)
output = model(**model_inputs)

def get_video_duration(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num/rate
        return duration
    return -1

duration = get_video_duration(file)
start, end = processor.post_process_video_grounding(output.logits, duration)

print(f"The time slot of the video corresponding to the text \"{text}\" is from {start}s to {end}s")
```

Tips:

* This implementation of TVP uses [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) to generate text embeddings and Resnet-50 model to compute visual embeddings.
* Checkpoints for pre-trained [tvp-base](https://huggingface.co/Intel/tvp-base) is released.
* Please refer to [Table 2](https://huggingface.co/papers/2303.04995) for TVP’s performance on Temporal Video Grounding task.

## TvpConfig

### class transformers.TvpConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/configuration_tvp.py#L28)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None distance\_loss\_weight = 1.0 duration\_loss\_weight = 0.1 visual\_prompter\_type = 'framepad' visual\_prompter\_apply = 'replace' visual\_prompt\_size = 96 max\_img\_size = 448 num\_frames = 48 vocab\_size = 30522 hidden\_size = 768 intermediate\_size = 3072 num\_hidden\_layers = 12 num\_attention\_heads = 12 max\_position\_embeddings = 512 max\_grid\_col\_position\_embeddings = 100 max\_grid\_row\_position\_embeddings = 100 hidden\_dropout\_prob = 0.1 hidden\_act = 'gelu' layer\_norm\_eps = 1e-12 initializer\_range = 0.02 attention\_probs\_dropout\_prob = 0.1 \*\*kwargs  )

Parameters

* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*) —
  The configuration of the backbone model.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **distance\_loss\_weight** (`float`, *optional*, defaults to 1.0) —
  The weight of distance loss.
* **duration\_loss\_weight** (`float`, *optional*, defaults to 0.1) —
  The weight of duration loss.
* **visual\_prompter\_type** (`str`, *optional*, defaults to `"framepad"`) —
  Visual prompt type. The type of padding. Framepad means padding on each frame. Should be one of “framepad”
  or “framedownpad”
* **visual\_prompter\_apply** (`str`, *optional*, defaults to `"replace"`) —
  The way of applying visual prompt. Replace means use the value of prompt to change the original value in
  visual inputs. Should be one of “replace”, or “add”, or “remove”.
* **visual\_prompt\_size** (`int`, *optional*, defaults to 96) —
  The size of visual prompt.
* **max\_img\_size** (`int`, *optional*, defaults to 448) —
  The maximum size of frame.
* **num\_frames** (`int`, *optional*, defaults to 48) —
  The number of frames extracted from a video.
* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Tvp text model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [TvpModel](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **max\_grid\_col\_position\_embeddings** (`int`, *optional*, defaults to 100) —
  The largest number of horizontal patches from a video frame.
* **max\_grid\_row\_position\_embeddings** (`int`, *optional*, defaults to 100) —
  The largest number of vertical patches from a video frame.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability of hidden layers.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability of attention layers.

This is the configuration class to store the configuration of a [TvpModel](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpModel). It is used to instantiate an Tvp
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Tvp
[Intel/tvp-base](https://huggingface.co/Intel/tvp-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_backbone\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/configuration_tvp.py#L183)

( backbone\_config: PretrainedConfig \*\*kwargs  ) → [TvpConfig](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpConfig)

Parameters

* **backbone\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The backbone configuration.

Returns

[TvpConfig](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpConfig)

An instance of a configuration object

Instantiate a [TvpConfig](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpConfig) (or a derived class) from a pre-trained backbone model configuration.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/configuration_tvp.py#L195)

( ) → `dict[str, any]`

Returns

`dict[str, any]`

Dictionary of all the attributes that make up this configuration instance,

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.to_dict).

## TvpImageProcessor

### class transformers.TvpImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/image_processing_tvp.py#L85)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_pad: bool = True pad\_size: typing.Optional[dict[str, int]] = None constant\_values: typing.Union[float, collections.abc.Iterable[float]] = 0 pad\_mode: PaddingMode = <PaddingMode.CONSTANT: 'constant'> do\_normalize: bool = True do\_flip\_channel\_order: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"longest_edge" -- 448}`):
  Size of the output image after resizing. The longest edge of the image will be resized to
  `size["longest_edge"]` while maintaining the aspect ratio of the original image. Can be overridden by
  `size` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
  parameter in the `preprocess` method.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 448, "width": 448}`):
  Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
  `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
  in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess` method.
* **pad\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 448, "width": 448}`):
  Size of the image after applying the padding. Can be overridden by the `pad_size` parameter in the
  `preprocess` method.
* **constant\_values** (`Union[float, Iterable[float]]`, *optional*, defaults to 0) —
  The fill value to use when padding the image.
* **pad\_mode** (`PaddingMode`, *optional*, defaults to `PaddingMode.CONSTANT`) —
  Use what kind of mode in padding.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **do\_flip\_channel\_order** (`bool`, *optional*, defaults to `True`) —
  Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
  parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a Tvp image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/image_processing_tvp.py#L340)

( videos: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], list[list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]]] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_pad: typing.Optional[bool] = None pad\_size: typing.Optional[dict[str, int]] = None constant\_values: typing.Union[float, collections.abc.Iterable[float], NoneType] = None pad\_mode: PaddingMode = None do\_normalize: typing.Optional[bool] = None do\_flip\_channel\_order: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **videos** (`ImageInput` or `list[ImageInput]` or `list[list[ImageInput]]`) —
  Frames to preprocess.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after applying resize.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_centre_crop`) —
  Whether to centre crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the image after applying the centre crop.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess` method.
* **pad\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 448, "width": 448}`):
  Size of the image after applying the padding. Can be overridden by the `pad_size` parameter in the
  `preprocess` method.
* **constant\_values** (`Union[float, Iterable[float]]`, *optional*, defaults to 0) —
  The fill value to use when padding the image.
* **pad\_mode** (`PaddingMode`, *optional*, defaults to “PaddingMode.CONSTANT”) —
  Use what kind of mode in padding.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **do\_flip\_channel\_order** (`bool`, *optional*, defaults to `self.do_flip_channel_order`) —
  Whether to flip the channel order of the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the inferred channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## TvpImageProcessorFast

### class transformers.TvpImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/image_processing_tvp_fast.py#L76)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.tvp.image\_processing\_tvp\_fast.TvpFastImageProcessorKwargs]  )

Constructs a fast Tvp image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/image_processing_tvp_fast.py#L98)

( videos: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], list[list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]]] \*\*kwargs: typing\_extensions.Unpack[transformers.models.tvp.image\_processing\_tvp\_fast.TvpFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **videos** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], list[list[Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]]]`) —
  Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
  passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
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
* **do\_flip\_channel\_order** (`bool`, *optional*) —
  Whether to flip the channel order of the image from RGB to BGR.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the image.
* **pad\_size** (`Dict[str, int]` or `SizeDict`, *optional*) —
  Size dictionary specifying the desired height and width for padding.
* **constant\_values** (`float` or `List[float]`, *optional*) —
  Value used to fill the padding area when `pad_mode` is `'constant'`.
* **pad\_mode** (`str`, *optional*) —
  Padding mode to use — `'constant'`, `'edge'`, `'reflect'`, or `'symmetric'`.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## TvpProcessor

### class transformers.TvpProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/processing_tvp.py#L23)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** ([TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast), *optional*) —
  The tokenizer is a required input.

Constructs an TVP processor which wraps a TVP image processor and a Bert tokenizer into a single processor.

[TvpProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor) offers all the functionalities of [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor) and [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast). See the
[**call**()](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/processing_tvp.py#L49)

( text = None videos = None return\_tensors = None \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **videos** (`list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, `list[list[PIL.Image.Image]]`, `list[list[np.ndarray]]`, —
  `list[list[torch.Tensor]]`): The video or batch of videos to be prepared. Each video should be a list
  of frames, which can be either PIL images or NumPy arrays. In case of NumPy arrays/PyTorch tensors,
  each frame should be of shape (H, W, C), where H and W are frame height and width, and C is a number of
  channels.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model. Returned when `text` is not `None`.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names` and if `text` is not
  `None`).
* **pixel\_values** — Pixel values to be fed to a model. Returned when `videos` is not `None`.

Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
and `kwargs` arguments to BertTokenizerFast’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None` to encode
the text. To prepare the image(s), this method forwards the `videos` and `kwargs` arguments to
TvpImageProcessor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) if `videos` is not `None`. Please refer to the docstring of
the above two methods for more information.

## TvpModel

### class transformers.TvpModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/modeling_tvp.py#L724)

( config  )

Parameters

* **config** ([TvpModel](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Tvp Model transformer outputting BaseModelOutputWithPooling object without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/modeling_tvp.py#L754)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor). See [TvpImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([TvpProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor) uses
  [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor) for processing images).
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

The [TvpModel](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import AutoConfig, AutoTokenizer, TvpModel

>>> model = TvpModel.from_pretrained("Jiqing/tiny-random-tvp")

>>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

>>> pixel_values = torch.rand(1, 1, 3, 448, 448)
>>> text_inputs = tokenizer("This is an example input", return_tensors="pt")
>>> output = model(text_inputs.input_ids, pixel_values, text_inputs.attention_mask)
```

## TvpForVideoGrounding

### class transformers.TvpForVideoGrounding

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/modeling_tvp.py#L847)

( config  )

Parameters

* **config** ([TvpForVideoGrounding](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpForVideoGrounding)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tvp Model with a video grounding head on top computing IoU, distance, and duration loss.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tvp/modeling_tvp.py#L856)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None labels: typing.Optional[tuple[torch.Tensor]] = None head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor). See [TvpImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([TvpProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor) uses
  [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor) for processing images).
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **labels** (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*) —
  The labels contains duration, start time, and end time of the video corresponding to the text.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

The [TvpForVideoGrounding](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpForVideoGrounding) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import AutoConfig, AutoTokenizer, TvpForVideoGrounding

>>> model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp")

>>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

>>> pixel_values = torch.rand(1, 1, 3, 448, 448)
>>> text_inputs = tokenizer("This is an example input", return_tensors="pt")
>>> output = model(text_inputs.input_ids, pixel_values, text_inputs.attention_mask)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/tvp.md)
