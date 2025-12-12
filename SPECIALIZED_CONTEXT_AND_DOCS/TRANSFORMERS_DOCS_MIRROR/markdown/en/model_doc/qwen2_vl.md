*This model was released on 2024-08-29 and added to Hugging Face Transformers on 2024-08-26.*

# Qwen2-VL

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![Tensor parallelism](https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white)

## Overview

The [Qwen2-VL](https://huggingface.co/papers/2409.12191) ([blog post](https://qwenlm.github.io/blog/qwen2-vl/)) model is a major update to [Qwen-VL](https://huggingface.co/papers/2308.12966) from the Qwen team at Alibaba Research.

The abstract from the blog is the following:

*This blog introduces Qwen2-VL, an advanced version of the Qwen-VL model that has undergone significant enhancements over the past year. Key improvements include enhanced image comprehension, advanced video understanding, integrated visual agent functionality, and expanded multilingual support. The model architecture has been optimized for handling arbitrary image resolutions through Naive Dynamic Resolution support and utilizes Multimodal Rotary Position Embedding (M-ROPE) to effectively process both 1D textual and multi-dimensional visual data. This updated model demonstrates competitive performance against leading AI systems like GPT-4o and Claude 3.5 Sonnet in vision-related tasks and ranks highly among open-source models in text capabilities. These advancements make Qwen2-VL a versatile tool for various applications requiring robust multimodal processing and reasoning abilities.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/qwen2_vl_architecture.jpeg) Qwen2-VL architecture. Taken from the [blog post.](https://qwenlm.github.io/blog/qwen2-vl/)

This model was contributed by [simonJJJ](https://huggingface.co/simonJJJ).

## Usage example

### Single Media inference

The model can accept both images and videos as input. Here’s an example code for inference.


```
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)



# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Batch Mixed Media Inference

The model can batch inputs composed of mixed samples of various types such as images, videos, and text. Here is an example.


```
# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image1.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image2.jpg"},
            {"type": "image", "path": "/path/to/image3.jpg"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "user",
        "content": "who are you?"
    }
]


# Conversation with mixed midia
conversation4 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image3.jpg"},
            {"type": "image", "path": "/path/to/image4.jpg"},
            {"type": "video", "path": "/path/to/video.jpg"},
            {"type": "text", "text": "What are the common elements in these medias?"},
        ],
    }
]

conversations = [conversation1, conversation2, conversation3, conversation4]
# Preparation for batch inference
ipnuts = processor.apply_chat_template(
    conversations,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Batch Inference
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Usage Tips

#### Image Resolution trade-off

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs.


```
min_pixels = 224*224
max_pixels = 2048*2048
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```

In case of limited GPU RAM, one can reduce the resolution as follows:


```
min_pixels = 256*28*28
max_pixels = 1024*28*28 
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```

This ensures each image gets encoded using a number between 256-1024 tokens. The 28 comes from the fact that the model uses a patch size of 14 and a temporal patch size of 2 (14 x 2 = 28).

#### Multiple Image Inputs

By default, images and video content are directly included in the conversation. When handling multiple images, it’s helpful to add labels to the images and videos for better reference. Users can control this behavior with the following settings:


```
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": "Hello, how are you?"}
        ]
    },
    {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. How can I assist you today?"
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you describe these images and video?"}, 
            {"type": "image"}, 
            {"type": "image"}, 
            {"type": "video"}, 
            {"type": "text", "text": "These are from my vacation."}
        ]
    },
    {
        "role": "assistant",
        "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
    },
    {
        "role": "user",
        "content": "It was a trip to the mountains. Can you see the details in the images and video?"
    }
]

# default:
prompt_without_id = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'


# add ids
prompt_with_id = processor.apply_chat_template(conversation, add_generation_prompt=True, add_vision_id=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?Picture 2: <|vision_start|><|image_pad|><|vision_end|>Picture 3: <|vision_start|><|image_pad|><|vision_end|>Video 1: <|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:


```
pip install -U flash-attn --no-build-isolation
```

Also, you should have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using Flash Attention-2, simply add `attn_implementation="flash_attention_2"` when loading the model as follows:


```
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)
```

## Qwen2VLConfig

### class transformers.Qwen2VLConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py#L262)

( text\_config = None vision\_config = None image\_token\_id = 151655 video\_token\_id = 151656 \*\*kwargs  )

Parameters

* **text\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLTextConfig`) —
  The config object or dictionary of the text backbone.
* **vision\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 151655) —
  The image token index to encode the image prompt.
* **video\_token\_id** (`int`, *optional*, defaults to 151656) —
  The video token index to encode the image prompt.

This is the configuration class to store the configuration of a [Qwen2VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLModel). It is used to instantiate a
Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

>>> # Initializing a Qwen2_5_VL style configuration
>>> configuration = Qwen2_5_VLConfig()

>>> # Initializing a model from the Qwen2-VL-7B style configuration
>>> model = Qwen2_5_VLForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2VLTextConfig

### class transformers.Qwen2VLTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py#L59)

( vocab\_size = 152064 hidden\_size = 8192 intermediate\_size = 29568 num\_hidden\_layers = 80 num\_attention\_heads = 64 num\_key\_value\_heads = 8 hidden\_act = 'silu' max\_position\_embeddings = 32768 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True tie\_word\_embeddings = False rope\_theta = 1000000.0 use\_sliding\_window = False sliding\_window = 4096 max\_window\_layers = 80 layer\_types = None attention\_dropout = 0.0 rope\_scaling = None image\_token\_id = None video\_token\_id = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 152064) —
  Vocabulary size of the Qwen2VL model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Qwen2VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLModel)
* **hidden\_size** (`int`, *optional*, defaults to 8192) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 29568) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 80) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 64) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
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
* **rope\_theta** (`float`, *optional*, defaults to 1000000.0) —
  The base period of the RoPE embeddings.
* **use\_sliding\_window** (`bool`, *optional*, defaults to `False`) —
  Whether to use sliding window attention.
* **sliding\_window** (`int`, *optional*, defaults to 4096) —
  Sliding window attention (SWA) window size. If not specified, will default to `4096`.
* **max\_window\_layers** (`int`, *optional*, defaults to 80) —
  The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
  additional layer afterwards will use SWA (Sliding Window Attention).
* **layer\_types** (`list`, *optional*) —
  Attention pattern for each layer.
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
  `attention_factor` (`float`,* optional*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
  `beta_fast` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.
  `beta_slow` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.
  `short_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **image\_token\_id** (`int`, *optional*) —
  Token index used as placeholder for image embeddings.
* **video\_token\_id** (`int`, *optional*) —
  Token index used as placeholder for video embeddings.

This is the configuration class to store the configuration of a [Qwen2VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextModel). It is used to instantiate a
Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Qwen2VLTextModel, Qwen2VLConfig

>>> # Initializing a Qwen2VL style configuration
>>> configuration = Qwen2VLConfig()

>>> # Initializing a model from the Qwen2-VL-7B style configuration
>>> model = Qwen2VLTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2VLImageProcessor

### class transformers.Qwen2VLImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True min\_pixels: typing.Optional[int] = None max\_pixels: typing.Optional[int] = None patch\_size: int = 14 temporal\_patch\_size: int = 2 merge\_size: int = 2 \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions.
* **size** (`dict[str, int]`, *optional*, defaults to `{"shortest_edge" -- 56 * 56, "longest_edge": 28 * 28 * 1280}`):
  Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use when resizing the image.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) —
  Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **min\_pixels** (`int`, *optional*, defaults to `56 * 56`) —
  The min pixels of the image to resize the image.
* **max\_pixels** (`int`, *optional*, defaults to `28 * 28 * 1280`) —
  The max pixels of the image to resize the image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The spatial patch size of the vision encoder.
* **temporal\_patch\_size** (`int`, *optional*, defaults to 2) —
  The temporal patch size of the vision encoder.
* **merge\_size** (`int`, *optional*, defaults to 2) —
  The merge size of the vision encoder to llm encoder.

Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L300)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None min\_pixels: typing.Optional[int] = None max\_pixels: typing.Optional[int] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None patch\_size: typing.Optional[int] = None temporal\_patch\_size: typing.Optional[int] = None merge\_size: typing.Optional[int] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **videos** (`VideoInput`) —
  Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
  passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
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
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **min\_pixels** (`int`, *optional*, defaults to `self.min_pixels`) —
  The min pixels of the image to resize the image.
* **max\_pixels** (`int`, *optional*, defaults to `self.max_pixels`) —
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

## Qwen2VLVideoProcessor

### class transformers.Qwen2VLVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py#L98)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.qwen2\_vl.video\_processing\_qwen2\_vl.Qwen2VLVideoProcessorInitKwargs]  )

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
* **min\_pixels** (`int`, *optional*, defaults to `56 * 56`) —
  The min pixels of the image to resize the image.
* **max\_pixels** (`int`, *optional*, defaults to `28 * 28 * 1280`) —
  The max pixels of the image to resize the image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The spacial patch size of the vision encoder.
* **temporal\_patch\_size** (`int`, *optional*, defaults to 2) —
  The temporal patch size of the vision encoder.
* **merge\_size** (`int`, *optional*, defaults to 2) —
  The merge size of the vision encoder to llm encoder.
* **min\_frames** (`int`, *optional*, defaults to 4) —
  The minimum number of frames that can be sampled.
* **max\_frames** (`int`, *optional*, defaults to 768) —
  The maximum number of frames that can be sampled.

Constructs a fast Qwen2-VL image processor that dynamically resizes videos based on the original videos.

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

## Qwen2VLImageProcessorFast

### class transformers.Qwen2VLImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py#L87)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.qwen2\_vl.image\_processing\_qwen2\_vl\_fast.Qwen2VLFastImageProcessorKwargs]  )

Constructs a fast Qwen2 Vl image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py#L144)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.qwen2\_vl.image\_processing\_qwen2\_vl\_fast.Qwen2VLFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **videos** (`Union[list['PIL.Image.Image'], np.ndarray, torch.Tensor, list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]], NoneType]`) —
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
* **min\_pixels** (`int`, *optional*, defaults to `56 * 56`) —
  The min pixels of the image to resize the image.
* **max\_pixels** (`int`, *optional*, defaults to `28 * 28 * 1280`) —
  The max pixels of the image to resize the image.
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

## Qwen2VLProcessor

### class transformers.Qwen2VLProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/processing_qwen2_vl.py#L57)

( image\_processor = None tokenizer = None video\_processor = None chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast), *optional*) —
  The tokenizer is a required input.
* **video\_processor** ([Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor), *optional*) —
  The video processor is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor and a Qwen2 tokenizer into a single processor.
[Qwen2VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) and [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### post\_process\_image\_text\_to\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/processing_qwen2_vl.py#L227)

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

## Qwen2VLTextModel

### class transformers.Qwen2VLTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L754)

( config: Qwen2VLTextConfig  )

Parameters

* **config** ([Qwen2VLTextConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen2 Vl Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L775)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig)) and inputs.

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

The [Qwen2VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen2VLModel

### class transformers.Qwen2VLModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L911)

( config: Qwen2VLConfig  )

Parameters

* **config** ([Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen2 Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1161)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModelOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Qwen2VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor) uses
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor). See `Qwen2VLVideoProcessor.__call__()` for details ([Qwen2VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor) uses
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) for processing videos).
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

`transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig)) and inputs.

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

The [Qwen2VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen2VLForConditionalGeneration

### class transformers.Qwen2VLForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1257)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1300)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Qwen2VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor) uses
  [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor). See `Qwen2VLVideoProcessor.__call__()` for details ([Qwen2VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor) uses
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) for processing videos).
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

`transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig)) and inputs.

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

The [Qwen2VLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

>>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md)
