*This model was released on 2024-05-31 and added to Hugging Face Transformers on 2024-06-26.*

# LLaVa-NeXT-Video

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The LLaVa-NeXT-Video model was proposed in [LLaVA-NeXT: A Strong Zero-shot Video Understanding Model](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) by Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, Chunyuan Li. LLaVa-NeXT-Video improves upon [LLaVa-NeXT](llava_next) by fine-tuning on a mix if video and image dataset thus increasing the model’s performance on videos.

[LLaVA-NeXT](llava_next) surprisingly has strong performance in understanding video content in zero-shot fashion with the AnyRes technique that it uses. The AnyRes technique naturally represents a high-resolution image into multiple images. This technique is naturally generalizable to represent videos because videos can be considered as a set of frames (similar to a set of images in LLaVa-NeXT). The current version of LLaVA-NeXT makes use of AnyRes and trains with supervised fine-tuning (SFT) on top of LLaVA-Next on video data to achieves better video understanding capabilities.The model is a current SOTA among open-source models on [VideoMME bench](https://huggingface.co/papers/2405.21075).

The introduction from the blog is the following:

On January 30, 2024, we released LLaVA-NeXT, an open-source Large Multimodal Model (LMM) that has been trained exclusively on text-image data. With the proposed AnyRes technique, it boosts capabilities in reasoning, OCR, and world knowledge, demonstrating remarkable performance across a spectrum of image-based multimodal understanding tasks, and even exceeding Gemini-Pro on several image benchmarks, e.g. MMMU and MathVista.

\*\*In today’s exploration, we delve into the performance of LLaVA-NeXT within the realm of video understanding tasks. We reveal that LLaVA-NeXT surprisingly has strong performance in understanding video content. The current version of LLaVA-NeXT for videos has several improvements:

* Zero-shot video representation capabilities with AnyRes: The AnyRes technique naturally represents a high-resolution image into multiple images that a pre-trained VIT is able to digest, and forms them into a concatenated sequence. This technique is naturally generalizable to represent videos (consisting of multiple frames), allowing the image-only-trained LLaVA-Next model to perform surprisingly well on video tasks. Notably, this is the first time that LMMs show strong zero-shot modality transfer ability.
* Inference with length generalization improves on longer videos. The linear scaling technique enables length generalization, allowing LLaVA-NeXT to effectively handle long-video beyond the limitation of the “max\_token\_length” of the LLM.
* Strong video understanding ability. (1) LLaVA-Next-Image, which combines the above two techniques, yields superior zero-shot performance than open-source LMMs tuned on videos. (2) LLaVA-Next-Video, further supervised fine-tuning (SFT) LLaVA-Next-Image on video data, achieves better video understanding capabilities compared to LLaVA-Next-Image. (3) LLaVA-Next-Video-DPO, which aligns the model response with AI feedback using direct preference optimization (DPO), showing significant performance boost.
* Efficient deployment and inference with SGLang. It allows 5x faster inference on video tasks, allowing more scalable serving such as million-level video re-captioning. See instructions in our repo.\*\*

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference).

## Usage tips

* We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.

* Llava-Next uses different number of patches for images and thus has to pad the inputs inside modeling code, aside from the padding done when processing the inputs. The default setting is “left-padding” if model is in `eval()` mode, otherwise “right-padding”.

> [!NOTE]
> LLaVA models after release v4.46 will raise warnings about adding `processor.patch_size = {{patch_size}}`, `processor.num_additional_image_tokens = {{num_additional_image_tokens}}` and processor.vision\_feature\_select\_strategy = {{vision\_feature\_select\_strategy}}`. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that LLaVA will try to infer the number of image tokens required per image and expand the text with as many` <image>`placeholders as there will be tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there will be failure when merging the embeddings. The attributes can be obtained from model config, as`model.config.vision\_config.patch\_size`or`model.config.vision\_feature\_select\_strategy`. The` num\_additional\_image\_tokens`should be`1`if the vision backbone adds a CLS token or`0` if nothing extra is added to the vision patches.

### Formatting Prompts with Chat Templates

Each **checkpoint** is trained with a specific prompt format, depending on the underlying large language model backbone. To ensure correct formatting, use the processor’s `apply_chat_template` method.

**Important:**

* You must construct a conversation history — passing a plain string won’t work.
* Each message should be a dictionary with `"role"` and `"content"` keys.
* The `"content"` should be a list of dictionaries for different modalities like `"text"` and `"image"`.

Here’s an example of how to structure your input. We will use [LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) and a conversation history of videos and images.


```
from transformers import LlavaNextVideoProcessor

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s shown in this image?"},
            {"type": "image"},
            ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video"},
            ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Note that the template simply formats your prompt, you still have to tokenize it and obtain pixel values for your visuals
print(text_prompt)
```

🚀 **Bonus:** If you’re using `transformers>=4.49.0`, you can also get a vectorized output from `apply_chat_template`. See the **Usage Examples** below for more details on how to use it.

## Usage example

### Single Media Mode

The model can accept both images and videos as input. Here’s an example code for inference in half-precision (`torch.float16`):


```
from huggingface_hub import hf_hub_download
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", dtype=torch.float16, device_map="auto")
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video", "path": video_path},
            ],
    },
]

inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=60)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```

### Mixed Media Mode

The model can also generate from an interleaved image-video inputs. However note, that it was not trained in interleaved image-video setting which might affect the performance. Below is an example usage for mixed media input, add the following lines to the above code snippet:


```
# Generate from image and video mixed inputs
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "How many cats are there in the image?"},
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            ],
    },
    {

        "role": "assistant",
        "content": [{"type": "text", "text": "There are two cats"}],
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video", "path": video_path},
            ],
    },
]
inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```

## Model optimization

### Quantization using Bitsandbytes for memory efficiency

The model can be loaded in lower bits, significantly reducing memory burden while maintaining the performance of the original model. This allows for efficient deployment on resource-constrained cases.

First, make sure to install bitsandbytes by running `pip install bitsandbytes` and to have access to a GPU/accelerator that is supported by the library.

bitsandbytes is being refactored to support multiple backends beyond CUDA. Currently, ROCm (AMD GPU) and Intel CPU implementations are mature, with Intel XPU in progress and Apple Silicon support expected by Q4/Q1. For installation instructions and the latest backend updates, visit [this link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend).

We value your feedback to help identify bugs before the full release! Check out [these docs](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends) for more details and feedback links.

Then simply load the quantized model by adding [`BitsAndBytesConfig`](../main_classes/quantization#transformers.BitsAndBytesConfig) as shown below:


```
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", quantization_config=quantization_config, device_map="auto")
```

### Flash-Attention 2 to speed-up generation

Additionally, we can greatly speed-up model inference by using [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2:


```
pip install -U flash-attn --no-build-isolation
```

Also, you should have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using Flash Attention-2, simply add `attn_implementation="flash_attention_2"` when loading the model as follows:


```
from transformers import LlavaNextVideoForConditionalGeneration

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", 
    dtype=torch.float16, 
    attn_implementation="flash_attention_2",
).to(0)
```

## LlavaNextVideoConfig

### class transformers.LlavaNextVideoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/configuration_llava_next_video.py#L26)

( vision\_config = None text\_config = None image\_token\_index = 32001 projector\_hidden\_act = 'gelu' multimodal\_projector\_bias = True vision\_feature\_select\_strategy = 'default' vision\_feature\_layer = -2 image\_grid\_pinpoints = None tie\_word\_embeddings = False video\_token\_index = 32000 spatial\_pool\_mode = 'average' spatial\_pool\_stride = 2 image\_seq\_length = 576 video\_seq\_length = 288 \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `CLIPVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`) —
  The config object or dictionary of the text backbone.
* **image\_token\_index** (`int`, *optional*, defaults to 32001) —
  The image token index to encode the image prompt.
* **projector\_hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The activation function used by the multimodal projector.
* **multimodal\_projector\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the multimodal projector.
* **vision\_feature\_select\_strategy** (`str`, *optional*, defaults to `"default"`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
  If `"full"`, the full vision features are used.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*, defaults to -2) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **image\_grid\_pinpoints** (`List`, *optional*, defaults to `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`) —
  A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
  of the form `(height, width)`.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **video\_token\_index** (`int`, *optional*, defaults to 32000) —
  The video token index to encode the image prompt.
* **spatial\_pool\_mode** (`str`, *optional*, defaults to `"average"`) —
  Pooling mode to use for videos. Can be “average”, “max” or “conv”.
* **spatial\_pool\_stride** (`int`, *optional*, defaults to 2) —
  Stride used in the pooling layer for videos.
* **image\_seq\_length** (`int`, *optional*, defaults to 576) —
  Sequence length of one image embedding.
* **video\_seq\_length** (`int`, *optional*, defaults to 288) —
  Sequence length of one video embedding.

This is the configuration class to store the configuration of a [LlavaNextVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoForConditionalGeneration). It is used to instantiate an
Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the [llava-hf/LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)
model.
Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoConfig, CLIPVisionConfig, LlamaConfig

>>> # Initializing a CLIP-vision config
>>> vision_config = CLIPVisionConfig()

>>> # Initializing a Llama config
>>> text_config = LlamaConfig()

>>> configuration = LlavaNextVideoConfig(vision_config, text_config)

>>> model = LlavaNextVideoForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LlavaNextVideoProcessor

### class transformers.LlavaNextVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/processing_llava_next_video.py#L47)

( video\_processor = None image\_processor = None tokenizer = None chat\_template = None patch\_size = None vision\_feature\_select\_strategy = None video\_token = '<video>' image\_token = '<image>' num\_additional\_image\_tokens = 0 \*\*kwargs  )

Parameters

* **video\_processor** ([LlavaNextVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoVideoProcessor), *optional*) —
  The video processor is a required input.
* **image\_processor** ([LlavaNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) —
  Jinja chat template that will be used in tokenizer’s `apply_chat_template`
* **patch\_size** (`int`, *optional*) —
  Patch size from the vision tower.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Should be same as in model’s config
* **video\_token** (`str`, *optional*, defaults to `"<video>"`) —
  Special token used to denote video location.
* **image\_token** (`str`, *optional*, defaults to `"<image>"`) —
  Special token used to denote image location.
* **num\_additional\_image\_tokens** (`int`, *optional*, defaults to 0) —
  Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
  extra tokens appended, no need to set this arg.

Constructs a LLaVa-NeXT-Video processor which wraps a LLaVa-NeXT image processor, LLaVa-NeXT-Video video processor and
a LLaMa tokenizer into a single processor.

[LlavaNextVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoProcessor) offers all the functionalities of [LlavaNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextImageProcessor), [LlavaNextVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoImageProcessor) and
[LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## LlavaNextVideoImageProcessor

### class transformers.LlavaNextVideoImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/image_processing_llava_next_video.py#L47)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None image\_grid\_pinpoints: typing.Optional[list] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True \*\*kwargs  )

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
* **do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **crop\_size** (`dict[str, int]` *optional*, defaults to 224) —
  Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
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
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a LLaVa-NeXT-Video video processor. Based on [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) with incorporation of processing each video frame.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/image_processing_llava_next_video.py#L278)

( images: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[int] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`VideoInput`) —
  Videos to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the video after resizing. Shortest edge of the video is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the video by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Frame mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Frame standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the video to RGB.
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

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/image_processing_llava_next_video.py#L128)

( image: ndarray size: dict resample: Resampling = <Resampling.BICUBIC: 3> data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Size of the output image.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  Resampling filter to use when resiizing the image.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the image. If not provided, it will be the same as the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.

Resize an image. The shortest edge of the image is resized to size[“shortest\_edge”], with the longest edge
resized to keep the input aspect ratio.

## LlavaNextVideoVideoProcessor

### class transformers.LlavaNextVideoVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/video_processing_llava_next_video.py#L37)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.llava\_next\_video.video\_processing\_llava\_next\_video.LlavaNextVideoFastVideoProcessorInitKwargs]  )

## LlavaNextVideoModel

### class transformers.LlavaNextVideoModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L304)

( config: LlavaNextVideoConfig  )

Parameters

* **config** ([LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig)) —
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L519)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None pixel\_values\_videos: FloatTensor = None image\_sizes: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LlavaNextVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoImageProcessor). See [LlavaNextVideoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LlavaNextVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoProcessor) uses
  [LlavaNextVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [LlavaNextVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoVideoProcessor). See `LlavaNextVideoVideoProcessor.__call__()` for details ([LlavaNextVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoProcessor) uses
  [LlavaNextVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoVideoProcessor) for processing videos).
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*) —
  The sizes of the images in the batch, being (height, width) for each image.
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
* **vision\_feature\_select\_strategy** (`str`, *optional*, defaults to `"default"`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
  If `"full"`, the full vision features are used.
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

`transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig)) and inputs.

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

The [LlavaNextVideoModel](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L403)

( pixel\_values: FloatTensor image\_sizes: Tensor vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None  ) → image\_features (list`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`) —
  The tensors corresponding to the input images.
* **image\_sizes** (`torch.Tensor` of shape `(num_images, 2)`) —
  Actual image size of each images (H, W).
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`

Returns

image\_features (list`torch.Tensor`)

List of image feature tensor, each contains all the visual feature of all patches
and are of shape `(num_patches, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L479)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor = None video\_features: FloatTensor = None  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

#### get\_video\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L618)

( pixel\_values: FloatTensor vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None  ) → video\_features (list`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, num_frames, channels, height, width)`) —
  The tensors corresponding to the input video.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optiona;*) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`

Returns

video\_features (list`torch.Tensor`)

List of video feature tensor, each contains all the visual feature of all patches
and are of shape `(num_videos, video_length, embed_dim)`).

Obtains video last hidden states from the vision tower and apply multimodal projection.

#### pack\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L336)

( image\_features image\_sizes vision\_feature\_select\_strategy image\_newline = None  )

Parameters

* **image\_features** (`list[torch.Tensor]` of length num\_images, each of shape `(num_patches, image_length, embed_dim)`) —
  List of image feature tensor, each contains all the visual feature of all patches.
* **image\_sizes** (`torch.Tensor` of shape `(num_images, 2)`) —
  Actual image size of each images (H, W).
* **vision\_feature\_select\_strategy** (`str`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
* **image\_newline** (`torch.Tensor` of shape `(embed_dim)`) —
  New line embedding vector.

Reshape, unpad and then pack each image\_feature into a single image\_features tensor containing all visual vectors.

## LlavaNextVideoForConditionalGeneration

### class transformers.LlavaNextVideoForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L679)

( config: LlavaNextVideoConfig  )

Parameters

* **config** ([LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig)) —
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L745)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None pixel\_values\_videos: FloatTensor = None image\_sizes: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LlavaNextVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoImageProcessor). See [LlavaNextVideoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LlavaNextVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoProcessor) uses
  [LlavaNextVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoImageProcessor) for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [LlavaNextVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoVideoProcessor). See `LlavaNextVideoVideoProcessor.__call__()` for details ([LlavaNextVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoProcessor) uses
  [LlavaNextVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoVideoProcessor) for processing videos).
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*) —
  The sizes of the images in the batch, being (height, width) for each image.
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

`transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig)) and inputs.

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

The [LlavaNextVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> import av
>>> from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

>>> model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", device_map="auto")
>>> processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

>>> prompt = "USER: <video>\nWhy is this video funny? ASSISTANT:"
>>> video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
>>> container = av.open(video_path)

>>> # sample uniformly 8 frames from the video (model was trained with 32 frames per video, but this video is short)
>>> total_frames = container.streams.video[0].frames
>>> indices = np.arange(0, total_frames, total_frames / 8).astype(int)
>>> clip = read_video_pyav(container, indices)
>>> inputs_video = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)

>>> # load an image to generate from an image
>>> prompt = "USER:<image>\nWhat is shown in this image? ASSISTANT:"
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs_image = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

>>> # Generate from video
>>> generate_ids = model.generate(**inputs_video, max_length=50)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"USER:\nWhy is this video funny? ASSISTANT: The humor in this video comes from the unexpected and endearing sight of a baby wearing glasses and (...)"

>>> # Generate from image
>>> generate_ids = model.generate(**inputs_image, max_length=30)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"USER: \nWhat's the content of the image? ASSISTANT: The image shows a red stop sign on a pole, with a traditional Chinese archway (...)"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llava_next_video.md)
