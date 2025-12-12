*This model was released on 2025-07-01 and added to Hugging Face Transformers on 2025-06-25.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# GLM-4.1V

## Overview

**GLM-4.1V-9B-Thinking** is a bilingual vision-language model optimized for reasoning, built on GLM-4-9B. It introduces
a “thinking paradigm” with reinforcement learning, achieving state-of-the-art results among 10B-class models and
rivaling 72B-scale models. It supports 64k context, 4K resolution, and arbitrary aspect ratios, with an open-source base
model for further research. You can check our paper [here](https://huggingface.co/papers/2507.01006). and below is a abstract.

*We present GLM-4.1V-Thinking, a vision-language model (VLM) designed to advance general-purpose multimodal understanding
and reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework.
We first develop a capable vision foundation model with significant potential through large-scale pre-training, which
arguably sets the upper bound for the final performance. We then propose Reinforcement Learning with Curriculum
Sampling (RLCS) to unlock the full potential of the model, leading to comprehensive capability enhancement across a
diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding,
GUI-based agents, and long document understanding. We open-source GLM-4.1V-9B-Thinking, which achieves state-of-the-art
performance among models of comparable size. In a comprehensive evaluation across 28 public benchmarks, our model
outperforms Qwen2.5-VL-7B on nearly all tasks and achieves comparable or even superior performance on 18 benchmarks
relative to the significantly larger Qwen2.5-VL-72B. Notably, GLM-4.1V-9B-Thinking also demonstrates competitive or
superior performance compared to closed-source models such as GPT-4o on challenging tasks including long document
understanding and STEM reasoning, further underscoring its strong capabilities. Code, models and more information
are released at <https://github.com/THUDM/GLM-4.1V-Thinking>.*

## Usage

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline
pipe = pipeline(
    task="image-text-to-text",
    model="THUDM/GLM-4.1V-9B-Thinking",
    device=0,
    dtype=torch.bfloat16
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe this image."},
        ]
    }
]
pipe(text=messages,max_new_tokens=20, return_full_text=False)
```

Using GLM-4.1V with video input is similar to using it with image input.
The model can process video data and generate text based on the content of the video.


```
from transformers import AutoProcessor, Glm4vForConditionalGeneration, infer_device
import torch

device = f"{infer_device()}:0"

processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="THUDM/GLM-4.1V-9B-Thinking",
    dtype=torch.bfloat16,
    device_map=device
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
            },
            {
                "type": "text",
                "text": "discribe this video",
            },
        ],
    }
]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(output_text)
```

## Glm4vConfig

### class transformers.Glm4vConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/configuration_glm4v.py#L275)

( text\_config = None vision\_config = None image\_token\_id = 151343 video\_token\_id = 151344 image\_start\_token\_id = 151339 image\_end\_token\_id = 151340 video\_start\_token\_id = 151341 video\_end\_token\_id = 151342 \*\*kwargs  )

Parameters

* **text\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vTextConfig`) —
  The config object or dictionary of the text backbone.
* **vision\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 151343) —
  The image token index to encode the image prompt.
* **video\_token\_id** (`int`, *optional*, defaults to 151344) —
  The video token index to encode the image prompt.
* **image\_start\_token\_id** (`int`, *optional*, defaults to 151339) —
  The image start token index to encode the start of image.
* **image\_end\_token\_id** (`int`, *optional*, defaults to 151340) —
  The image end token index to encode the end of image.
* **video\_start\_token\_id** (`int`, *optional*, defaults to 151341) —
  The video start token index to encode the start of video.
* **video\_end\_token\_id** (`int`, *optional*, defaults to 151342) —
  The video end token index to encode the end of video.

This is the configuration class to store the configuration of a [Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel). It is used to instantiate a
GLM-4.1V model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of
GLM-4.1V-9B-Thinking [THUDM/GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Glm4vForConditionalGeneration, Glm4vConfig

>>> # Initializing a GLM-4.1V style configuration
>>> configuration = Glm4vConfig()

>>> # Initializing a model from the GLM-4.1V style configuration
>>> model = Glm4vForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Glm4vTextConfig

### class transformers.Glm4vTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/configuration_glm4v.py#L122)

( vocab\_size = 151552 hidden\_size = 4096 intermediate\_size = 13696 num\_hidden\_layers = 40 num\_attention\_heads = 32 num\_key\_value\_heads = 2 hidden\_act = 'silu' max\_position\_embeddings = 32768 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True tie\_word\_embeddings = False rope\_theta = 10000.0 attention\_dropout = 0.0 rope\_scaling = None image\_token\_id = None video\_token\_id = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 151552) —
  Vocabulary size of the Glm4v model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 13696) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 40) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 2) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details checkout [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 32768) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *original maximum pre-trained length.
  `original_max_position_embeddings` (`int`,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`,* optional\*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
* **image\_token\_id** (`int`, *optional*) —
  Token index used as placeholder for image embeddings.
* **video\_token\_id** (`int`, *optional*) —
  Token index used as placeholder for video embeddings.

This is the configuration class to store the configuration of a [Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel). It is used to instantiate a
GLM-4.1V model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of
GLM-4.1V-9B-Thinking [THUDM/GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Glm4vTextModel, Glm4vConfig

>>> # Initializing a GLM-4.1V style configuration
>>> configuration = Glm4vConfig()

>>> # Initializing a model from the GLM-4.1V style configuration
>>> model = Glm4vTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Glm4vImageProcessor

### class transformers.Glm4vImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/image_processing_glm4v.py#L83)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True patch\_size: int = 14 temporal\_patch\_size: int = 2 merge\_size: int = 2 \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions.
* **size** (`Dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 112 * 112, "longest_edge": 28 * 28 * 15000}`):
  Size of the image’s `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
  in the `preprocess` method. Available options are:
  + `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
    Do NOT keep the aspect ratio.
  + `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
    the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
    less or equal to `longest_edge`.
  + `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
    aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
    `max_width`.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use when resizing the image.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) —
  Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
* **image\_std** (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The spatial patch size of the vision encoder.
* **temporal\_patch\_size** (`int`, *optional*, defaults to 2) —
  The temporal patch size of the vision encoder.
* **merge\_size** (`int`, *optional*, defaults to 2) —
  The merge size of the vision encoder to llm encoder.

Constructs a GLM-4V image processor that dynamically resizes images based on the original images.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/image_processing_glm4v.py#L297)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None patch\_size: typing.Optional[int] = None temporal\_patch\_size: typing.Optional[int] = None merge\_size: typing.Optional[int] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **videos** (`VideoInput`) —
  Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
  passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`Dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. Shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `List[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `List[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
  The max pixels of the image to resize the image.
* **patch\_size** (`int`, *optional*, defaults to `self.patch_size`) —
  The spatial patch size of the vision encoder.
* **temporal\_patch\_size** (`int`, *optional*, defaults to `self.temporal_patch_size`) —
  The temporal patch size of the vision encoder.
* **merge\_size** (`int`, *optional*, defaults to `self.merge_size`) —
  The merge size of the vision encoder to llm encoder.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

## Glm4vVideoProcessor

### class transformers.Glm4vVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/video_processing_glm4v.py#L79)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.glm4v.video\_processing\_glm4v.Glm4vVideoProcessorInitKwargs]  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `self.size`) —
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  The size by which to make sure both the height and width can be divided.
* **default\_to\_square** (`bool`, *optional*, defaults to `self.default_to_square`) —
  Whether to default to a square video when resizing, if size is an int.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) —
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.image_std`) —
  Whether to convert the video to RGB.
* **video\_metadata** (`VideoMetadata`, *optional*) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **do\_sample\_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) —
  Whether to sample frames from the video before processing or to process the whole video.
* **num\_frames** (`int`, *optional*, defaults to `self.num_frames`) —
  Maximum number of frames to sample when `do_sample_frames=True`.
* **fps** (`int` or `float`, *optional*, defaults to `self.fps`) —
  Target frames to sample per second when `do_sample_frames=True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input video.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the videos on. If unset, the device is inferred from the input videos.
* **return\_metadata** (`bool`, *optional*) —
  Whether to return video metadata or not.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The spacial patch size of the vision encoder.
* **temporal\_patch\_size** (`int`, *optional*, defaults to 2) —
  The temporal patch size of the vision encoder.
* **merge\_size** (`int`, *optional*, defaults to 2) —
  The merge size of the vision encoder to llm encoder.

Constructs a fast GLM-4V image processor that dynamically resizes videos based on the original videos.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L355)

( videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.VideosKwargs]  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `self.size`) —
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  The size by which to make sure both the height and width can be divided.
* **default\_to\_square** (`bool`, *optional*, defaults to `self.default_to_square`) —
  Whether to default to a square video when resizing, if size is an int.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) —
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.image_std`) —
  Whether to convert the video to RGB.
* **video\_metadata** (`VideoMetadata`, *optional*) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **do\_sample\_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) —
  Whether to sample frames from the video before processing or to process the whole video.
* **num\_frames** (`int`, *optional*, defaults to `self.num_frames`) —
  Maximum number of frames to sample when `do_sample_frames=True`.
* **fps** (`int` or `float`, *optional*, defaults to `self.fps`) —
  Target frames to sample per second when `do_sample_frames=True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input video.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the videos on. If unset, the device is inferred from the input videos.
* **return\_metadata** (`bool`, *optional*) —
  Whether to return video metadata or not.

## Glm4vImageProcessorFast

### class transformers.Glm4vImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/image_processing_glm4v_fast.py#L73)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.glm4v.image\_processing\_glm4v\_fast.Glm4vFastImageProcessorKwargs]  )

Constructs a fast Glm4V image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/image_processing_glm4v_fast.py#L196)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.glm4v.image\_processing\_glm4v\_fast.Glm4vFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The spatial patch size of the vision encoder.
* **temporal\_patch\_size** (`int`, *optional*, defaults to 2) —
  The temporal patch size of the vision encoder.
* **merge\_size** (`int`, *optional*, defaults to 2) —
  The merge size of the vision encoder to llm encoder.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Glm4vProcessor

### class transformers.Glm4vProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/processing_glm4v.py#L58)

( image\_processor = None tokenizer = None video\_processor = None chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **video\_processor** ([Glm4vVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vVideoProcessor), *optional*) —
  The video processor is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a GLM-4V processor which wraps a GLM-4V image processor and a GLM-4 tokenizer into a single processor.
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### post\_process\_image\_text\_to\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/processing_glm4v.py#L266)

( generated\_outputs skip\_special\_tokens = True clean\_up\_tokenization\_spaces = False \*\*kwargs  ) → `list[str]`

Parameters

* **generated\_outputs** (`torch.Tensor` or `np.ndarray`) —
  The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
  or `(sequence_length,)`.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to remove special tokens in the output. Argument passed to the tokenizer’s `batch_decode` method.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `False`) —
  Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer’s `batch_decode` method.
* \***\*kwargs** —
  Additional arguments to be passed to the tokenizer’s `batch_decode method`.

Returns

`list[str]`

The decoded text.

Post-process the output of the model to decode the text.

## Glm4vTextModel

### class transformers.Glm4vTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/modeling_glm4v.py#L804)

( config: Glm4vTextConfig  )

Parameters

* **config** ([Glm4vTextConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Glm4V Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/modeling_glm4v.py#L823)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Glm4vTextModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Glm4vModel

### class transformers.Glm4vModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/modeling_glm4v.py#L893)

( config  )

Parameters

* **config** ([Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Glm4V Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/modeling_glm4v.py#L1193)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.glm4v.modeling_glm4v.Glm4vModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  `video_processor_class`. See `video_processor_class.__call__` for details (`processor_class` uses
  `video_processor_class` for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.glm4v.modeling_glm4v.Glm4vModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.glm4v.modeling_glm4v.Glm4vModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Glm4vForConditionalGeneration

### class transformers.Glm4vForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/modeling_glm4v.py#L1331)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v/modeling_glm4v.py#L1373)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.glm4v.modeling_glm4v.Glm4vCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Glm4vImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vImageProcessor). See [Glm4vImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor) uses
  [Glm4vImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Glm4vVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vVideoProcessor). See `Glm4vVideoProcessor.__call__()` for details ([Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor) uses
  [Glm4vVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vVideoProcessor) for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.glm4v.modeling_glm4v.Glm4vCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.glm4v.modeling_glm4v.Glm4vCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Glm4vConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Glm4vForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Glm4vForConditionalGeneration

>>> model = Glm4vForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
>>> processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")

>>> messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
>>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm4v.md)
