*This model was released on 2024-08-06 and added to Hugging Face Transformers on 2024-09-05.*

# LLaVA-OneVision

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The LLaVA-OneVision model was proposed in [LLaVA-OneVision: Easy Visual Task Transfer](https://huggingface.co/papers/2408.03326) by <Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li

LLaVA-OneVision is a Vision-Language Model that can generate text conditioned on one or several images/videos. The model consists of SigLIP vision encoder and a Qwen2 language backbone. The images are processed with anyres-9 technique where the image is split into 9 patches to better process high resolution images and capture as much details as possible. However, videos are pooled to a total sequence length of 196 tokens each frame for more memory efficient computation. LLaVA-OneVision is available in three sizes: 0.5B, 7B and 72B and achieves remarkable performance on benchmark evaluations.

The abstract from the paper is the following:

*We present LLaVA-OneVision, a family of open large multimodal models (LMMs)
developed by consolidating our insights into data, models, and visual representations in the LLaVA-NeXT blog series. Our experimental results demonstrate that
LLaVA-OneVision is the first single model that can simultaneously push the performance boundaries of open LMMs in three important computer vision scenarios:
single-image, multi-image, and video scenarios. Importantly, the design of LLaVAOneVision allows strong transfer learning across different modalities/scenarios,
yielding new emerging capabilities. In particular, strong video understanding and
cross-scenario capabilities are demonstrated through task transfer from images to
videos.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava-ov-architecture.png) LLaVA-OneVision architecture. Taken from the [original paper.](https://huggingface.co/papers/2408.03326)

Tips:

* We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.

* Llava-OneVision uses different number of patches for images and thus has to pad the inputs inside modeling code, aside from the padding done when processing the inputs. The default setting is “left-padding” if model is in `eval()` mode, otherwise “right-padding”.

### Formatting Prompts with Chat Templates

Each **checkpoint** is trained with a specific prompt format, depending on the underlying large language model backbone. To ensure correct formatting, use the processor’s `apply_chat_template` method.

**Important:**

* You must construct a conversation history — passing a plain string won’t work.
* Each message should be a dictionary with `"role"` and `"content"` keys.
* The `"content"` should be a list of dictionaries for different modalities like `"text"` and `"image"`.

Here’s an example of how to structure your input.
We will use [llava-onevision-qwen2-7b-si-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-si-hf) and a conversation history of text and image. Each content field has to be a list of dicts, as follows:


```
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-si-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What’s shown in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in more details."},
        ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Note that the template simply formats your prompt, you still have to tokenize it and obtain pixel values for your images
print(text_prompt)
'<|im_start|>user\n<image>What is shown in this image?<|im_end|>\n<|im_start|>assistant\nPage showing the list of options.<|im_end|>'
```

🚀 **Bonus:** If you’re using `transformers>=4.49.0`, you can also get a vectorized output from `apply_chat_template`. See the **Usage Examples** below for more details on how to use it.

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main).

## Usage example

### Single image inference

Here’s how to load the model and perform inference in half-precision (`torch.float16`):


```
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, infer_device
import torch

device = f"{infer_device}:0"

processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    dtype=torch.float16,
    device_map=device
)

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device, torch.float16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
'user\n\nWhat is shown in this image?\nassistant\nThe image shows a radar chart, also known as a spider chart or a star chart, which is used to compare multiple quantitative variables. Each axis represents a different variable, and the chart is filled with'
```

### Multi image inference

LLaVa-OneVision can perform inference with multiple images as input, where images either belong to the same prompt or different prompts (in batched inference). For that you have to use checkpoints with an “ov” suffix. For multi-image cases, we recommend using a **nested list of images** as input. Otherwise, every image will be patchified and consume a lot of memory. Here is how you can do it:


```
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Load the model in half-precision
model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

# Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There is a red stop sign in the image."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What about this image? How many cats do you see?"},
        ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    [conversation_1, conversation_2],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    padding=True,
    padding_side="left",
    return_tensors="pt",
).to(model.device, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
['user\n\nWhat is shown in this image?\nassistant\nThere is a red stop sign in the image.\nuser\n\nWhat about this image? How many cats do you see?\nassistant\ntwo', 'user\n\nWhat is shown in this image?\nassistant\nThe image shows a whimsical scene of a snowman sitting by a campfire. The snowman is anthropomorphized, wearing a hat and']
```

### Video inference

LLaVa-OneVision also can perform inference with videos as input, where video frames are treated as multiple images. Here is how you can do it:


```
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Load the model in half-precision
model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Why is this video funny?"},
            ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    num_frames=8
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.float16)

out = model.generate(**inputs, max_new_tokens=60)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
["user\n\nWhy is this video funny?\nassistant\nThe video appears to be humorous because it shows a young child, who is wearing glasses and holding a book, seemingly reading with a serious and focused expression. The child's glasses are a bit oversized for their face, which adds a comical touch, as it's a common trope to see children wearing"]
```

## Model optimization

### Quantization using bitsandbytes

The model can be loaded in 8 or 4 bits, greatly reducing the memory requirements while maintaining the performance of the original model. First make sure to install bitsandbytes, `pip install bitsandbytes` and make sure to have access to a GPU/accelerator that is supported by the library.

bitsandbytes is being refactored to support multiple backends beyond CUDA. Currently, ROCm (AMD GPU) and Intel CPU implementations are mature, with Intel XPU in progress and Apple Silicon support expected by Q4/Q1. For installation instructions and the latest backend updates, visit [this link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend).

We value your feedback to help identify bugs before the full release! Check out [these docs](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends) for more details and feedback links.

Simply change the snippet above with:


```
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
```

### Use Flash-Attention 2 to further speed-up generation

First make sure to install flash-attn. Refer to the [original repository of Flash Attention](https://github.com/Dao-AILab/flash-attention) regarding that package installation. Simply change the snippet above with:


```
from transformers import LlavaOnevisionForConditionalGeneration

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_flash_attention_2=True
).to(0)
```

## LlavaOnevisionConfig

### class transformers.LlavaOnevisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/configuration_llava_onevision.py#L27)

( vision\_config = None text\_config = None image\_token\_index = 151646 video\_token\_index = 151647 projector\_hidden\_act = 'gelu' vision\_feature\_select\_strategy = 'full' vision\_feature\_layer = -1 vision\_aspect\_ratio = 'anyres\_max\_9' image\_grid\_pinpoints = None tie\_word\_embeddings = False multimodal\_projector\_bias = True \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `SiglipVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`) —
  The config object or dictionary of the text backbone.
* **image\_token\_index** (`int`, *optional*, defaults to 151646) —
  The image token index to encode the image prompt.
* **video\_token\_index** (`int`, *optional*, defaults to 151647) —
  The video token index to encode the video prompt.
* **projector\_hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The activation function used by the multimodal projector.
* **vision\_feature\_select\_strategy** (`str`, *optional*, defaults to `"full"`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
  If `"full"`, the full vision features are used.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*, defaults to -1) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_aspect\_ratio** (`str`, *optional*, defaults to `"anyres_max_9"`) —
  Aspect ratio used when processong image features. The default value is “anyres\_max\_9”.
* **image\_grid\_pinpoints** (`List`, *optional*) —
  A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
  of the form `(height, width)`.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **multimodal\_projector\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the multimodal projector.

This is the configuration class to store the configuration of a [LlavaOnevisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionForConditionalGeneration). It is used to instantiate an
Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
model.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig, SiglipVisionConfig, Qwen2Config

>>> # Initializing a CLIP-vision config
>>> vision_config = SiglipVisionConfig()

>>> # Initializing a Llama config
>>> text_config = Qwen2Config()

>>> # Initializing a Llava-Next llava-hf/llava-onevision-qwen2-7b-ov-hf style configuration
>>> configuration = LlavaOnevisionConfig(vision_config, text_config)

>>> # Initializing a model from the llava-hf/llava-onevision-qwen2-7b-ov-hf style configuration
>>> model = LlavaOnevisionForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LlavaOnevisionProcessor

### class transformers.LlavaOnevisionProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/processing_llava_onevision.py#L49)

( image\_processor = None tokenizer = None video\_processor = None num\_image\_tokens = None vision\_feature\_select\_strategy = None chat\_template = None image\_token = '<image>' video\_token = '<video>' vision\_aspect\_ratio = 'anyres\_max\_9' \*\*kwargs  )

Parameters

* **image\_processor** ([LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **video\_processor** ([LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor), *optional*) —
  The video processor is a required input.
* **num\_image\_tokens** (`int`, *optional*) —
  Number of image tokens for one imagethat will be returned by vision tower.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Should be same as in model’s config
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.
* **image\_token** (`str`, *optional*, defaults to `"<image>"`) —
  Special token used to denote image location.
* **video\_token** (`str`, *optional*, defaults to `"<video>"`) —
  Special token used to denote video location.
* **vision\_aspect\_ratio** (`str`, *optional*, defaults to `"anyres_max_9"`) —
  Aspect ratio used when processong image features. The default value is “anyres\_max\_9”.

Constructs a LLaVa-Onevision processor which wraps a LLaVa-Onevision video processor, LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

[LlavaNextProcessor](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextProcessor) offers all the functionalities of [LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor), [LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See the
`__call__()`, `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## LlavaOnevisionImageProcessor

### class transformers.LlavaOnevisionImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/image_processing_llava_onevision.py#L108)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None image\_grid\_pinpoints: typing.Optional[list] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = True do\_convert\_rgb: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by
  `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 224}`):
  Size of the image after resizing. The shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
  method.
* **image\_grid\_pinpoints** (`List` *optional*, defaults to `[[672, 336], [336, 672], [672, 672], [336, 1008], [1008, 336]]`) —
  A list of possible resolutions to use for processing high resolution images. The best resolution is selected
  based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
  method. Not used for processing videos.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a LLaVa-Onevision image processor. Based on [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) with incorporation of processing each video frame.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/image_processing_llava_onevision.py#L600)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None image\_grid\_pinpoints: typing.Optional[list] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. Both channels-first and channels-last formats are supported.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. Shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio.
* **image\_grid\_pinpoints** (`List` *optional*, defaults to `self.image_grid_pinpoints`) —
  A list of possible resolutions to use for processing high resolution images. The best resolution is
  selected based on the original size of the image.
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
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
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

## LlavaOnevisionImageProcessorFast

### class transformers.LlavaOnevisionImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py#L69)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.llava\_onevision.image\_processing\_llava\_onevision\_fast.LlavaOnevisionFastImageProcessorKwargs]  )

Constructs a fast Llava Onevision image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py#L89)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.llava\_onevision.image\_processing\_llava\_onevision\_fast.LlavaOnevisionFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **image\_grid\_pinpoints** (`list[list[int]]`, *optional*) —
  A list of possible resolutions to use for processing high resolution images. The best resolution is selected
  based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
  method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## LlavaOnevisionVideoProcessor

### class transformers.LlavaOnevisionVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/video_processing_llava_onevision.py#L37)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.llava\_onevision.video\_processing\_llava\_onevision.LlavaOnevisionFastVideoProcessorInitKwargs]  )

## LlavaOnevisionModel

### class transformers.LlavaOnevisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L271)

( config  )

Parameters

* **config** ([LlavaOnevisionModel](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Llava-Next model which consists of a vision backbone and a language model without language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L494)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None image\_sizes: typing.Optional[torch.LongTensor] = None pixel\_values\_videos: FloatTensor = None image\_sizes\_videos: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None vision\_aspect\_ratio: typing.Optional[str] = None batch\_num\_images: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor). See [LlavaOnevisionImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LlavaOnevisionProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionProcessor) uses
  [LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor) for processing images).
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*) —
  The sizes of the images in the batch, being (height, width) for each image.
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor). See `LlavaOnevisionVideoProcessor.__call__()` for details ([LlavaOnevisionProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionProcessor) uses
  [LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor) for processing videos).
* **image\_sizes\_videos** (`torch.LongTensor` of shape `(batch_size, frames, 2)`, *optional*) —
  The sizes of the videos in the batch, being (height, width) for each frame in the video.
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
* **vision\_feature\_layer** (`Union[int, list[int], NoneType]`) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.
* **vision\_aspect\_ratio** (`str`, *optional*, defaults to `"anyres_max_9"`) —
  Aspect ratio used when processong image features. The default value is “anyres\_max\_9”.
* **batch\_num\_images** (`torch.LongTensor`, *optional*) —
  Number of images in each sample.
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

`transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.
* **video\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size * num_frames, num_videos, sequence_length, hidden_size)`.
  video\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [LlavaOnevisionModel](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L364)

( pixel\_values: FloatTensor image\_sizes: Tensor vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None vision\_aspect\_ratio: typing.Optional[str] = None batch\_num\_images: typing.Optional[torch.LongTensor] = None  ) → image\_features (list`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`) —
  The tensors corresponding to the input images.
* **image\_sizes** (`torch.Tensor` of shape `(num_images, 2)`) —
  Actual image size of each images (H, W).
* **vision\_feature\_layer** (`Union[int, list[int]]`) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`
* **batch\_num\_images** (`torch.LongTensor`, *optional*) —
  Number of images in each sample.

Returns

image\_features (list`torch.Tensor`)

List of image feature tensor, each contains all the visual feature of all patches
and are of shape `(num_patches, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L454)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor = None video\_features: FloatTensor = None  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

#### get\_video\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L605)

( pixel\_values: FloatTensor vision\_feature\_layer: typing.Union[int, list[int]] vision\_feature\_select\_strategy: str  ) → video\_features (list`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, num_frames, channels, height, width)`) —
  The tensors corresponding to the input video.
* **vision\_feature\_layer** (`Union[int, list[int]], *optional*, defaults to -2`) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`

Returns

video\_features (list`torch.Tensor`)

List of video feature tensor, each contains all the visual feature of all patches
and are of shape `(num_videos, video_length, embed_dim)`).

Obtains video last hidden states from the vision tower, apply multimodal projection and pooling.

#### pack\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L299)

( image\_features image\_sizes image\_newline = None vision\_aspect\_ratio = 'anyres\_max\_9'  )

Parameters

* **image\_features** (`list[torch.Tensor]` of length num\_images, each of shape `(num_patches, image_length, embed_dim)`) —
  List of image feature tensor, each contains all the visual feature of all patches.
* **image\_sizes** (`torch.Tensor` of shape `(num_images, 2)`) —
  Actual image size of each images (H, W).
* **image\_newline** (`torch.Tensor` of shape `(embed_dim)`) —
  New line embedding vector.
* **vision\_aspect\_ratio** (`str`, *optional*, “anyres\_max\_9”) —
  Aspect ratio used when processong image features. The default value is “anyres\_max\_9”.

Reshape, unpad and then pack each image\_feature into a single image\_features tensor containing all visual vectors.

## LlavaOnevisionForConditionalGeneration

### class transformers.LlavaOnevisionForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L671)

( config: LlavaOnevisionConfig  )

Parameters

* **config** ([LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LLAVA-NeXT model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L737)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None image\_sizes: typing.Optional[torch.LongTensor] = None pixel\_values\_videos: FloatTensor = None image\_sizes\_videos: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None vision\_aspect\_ratio: typing.Optional[str] = None batch\_num\_images: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor). See [LlavaOnevisionImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LlavaOnevisionProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionProcessor) uses
  [LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor) for processing images).
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*) —
  The sizes of the images in the batch, being (height, width) for each image.
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor). See `LlavaOnevisionVideoProcessor.__call__()` for details ([LlavaOnevisionProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionProcessor) uses
  [LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor) for processing videos).
* **image\_sizes\_videos** (`torch.LongTensor` of shape `(batch_size, frames, 2)`, *optional*) —
  The sizes of the videos in the batch, being (height, width) for each frame in the video.
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
* **vision\_feature\_layer** (`Union[int, list[int], NoneType]`) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.
* **vision\_aspect\_ratio** (`str`, *optional*, defaults to `"anyres_max_9"`) —
  Aspect ratio used when processong image features. The default value is “anyres\_max\_9”.
* **batch\_num\_images** (`torch.LongTensor`, *optional*) —
  Number of images in each sample.
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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
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

`transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig)) and inputs.

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
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size (batch\_size \* num\_patches, num\_images, sequence\_length, hidden\_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.
* **video\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size * num_frames, num_videos, sequence_length, hidden_size)`.
  video\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [LlavaOnevisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> import torch
>>> from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

>>> model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype="float16", device_map="cuda:0")
>>> processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

>>> conversation = [
...     {
...       "role": "user",
...       "content": [
...           {"type": "text", "text": "What is shown in this image?"},
...           {"type": "image"},
...         ],
...     },
... ]
>>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

>>> image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> raw_image = Image.open(requests.get(image_file, stream=True).raw)
>>> inputs = processor(text=prompt, images=raw_image, return_tensors='pt').to(0, torch.float16)

>>> output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
>>> processor.batch_decode(output, skip_special_tokens=True)[0]
"user\n\nWhat is shown in this image?\nassistant\ncat"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llava_onevision.md)
