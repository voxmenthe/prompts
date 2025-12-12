*This model was released on 2023-04-17 and added to Hugging Face Transformers on 2023-12-07.*

# LLaVa

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

LLaVa is an open-source chatbot trained by fine-tuning LlamA/Vicuna on GPT-generated multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture. In other words, it is an multi-modal version of LLMs fine-tuned for chat / instructions.

The LLaVa model was proposed in [Visual Instruction Tuning](https://huggingface.co/papers/2304.08485) and improved in [Improved Baselines with Visual Instruction Tuning](https://huggingface.co/papers/2310.03744) by Haotian Liu, Chunyuan Li, Yuheng Li and Yong Jae Lee.

The abstract from the paper is the following:

*Large multimodal models (LMM) have recently shown encouraging progress with visual instruction tuning. In this note, we show that the fully-connected vision-language cross-modal connector in LLaVA is surprisingly powerful and data-efficient. With simple modifications to LLaVA, namely, using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with simple response formatting prompts, we establish stronger baselines that achieve state-of-the-art across 11 benchmarks. Our final 13B checkpoint uses merely 1.2M publicly available data, and finishes full training in ∼1 day on a single 8-A100 node. We hope this can make state-of-the-art LMM research more accessible. Code and model will be publicly available*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_architecture.jpg) LLaVa architecture. Taken from the [original paper.](https://huggingface.co/papers/2304.08485)

This model was contributed by [ArthurZ](https://huggingface.co/ArthurZ) and [ybelkada](https://huggingface.co/ybelkada).
The original code can be found [here](https://github.com/haotian-liu/LLaVA/tree/main/llava).

## Usage tips

* We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.
* Note the model has not been explicitly trained to process multiple images in the same prompt, although this is technically possible, you may experience inaccurate results.

> [!NOTE]
> LLaVA models after release v4.46 will raise warnings about adding `processor.patch_size = {{patch_size}}`, `processor.num_additional_image_tokens = {{num_additional_image_tokens}}` and processor.vision\_feature\_select\_strategy = {{vision\_feature\_select\_strategy}}`. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that LLaVA will try to infer the number of image tokens required per image and expand the text with as many` <image>`placeholders as there will be tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there will be failure when merging the embeddings. The attributes can be obtained from model config, as`model.config.vision\_config.patch\_size`or`model.config.vision\_feature\_select\_strategy`. The` num\_additional\_image\_tokens`should be`1`if the vision backbone adds a CLS token or`0` if nothing extra is added to the vision patches.

### Formatting Prompts with Chat Templates

Each **checkpoint** is trained with a specific prompt format, depending on the underlying large language model backbone. To ensure correct formatting, use the processor’s `apply_chat_template` method.

**Important:**

* You must construct a conversation history — passing a plain string won’t work.
* Each message should be a dictionary with `"role"` and `"content"` keys.
* The `"content"` should be a list of dictionaries for different modalities like `"text"` and `"image"`.

Here’s an example of how to structure your input.
We will use [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and a conversation history of text and image. Each content field has to be a list of dicts, as follows:


```
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

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
>>> "USER: <image>\n<What’s shown in this image? ASSISTANT: This image shows a red stop sign.</s>USER: Describe the image in more details. ASSISTANT:"
```

* If you want to construct a chat prompt yourself, below is a list of prompt formats accepted by each llava checkpoint:

[llava-interleave models](https://huggingface.co/collections/llava-hf/llava-interleave-668e19a97da0036aad4a2f19) requires the following format:


```
"<|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant"
```

For multiple turns conversation:


```
"<|im_start|>user <image>\n<prompt1><|im_end|><|im_start|>assistant <answer1><|im_end|><|im_start|>user <image>\n<prompt1><|im_end|><|im_start|>assistant "
```

[llava-1.5 models](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0) requires the following format:


```
"USER: <image>\n<prompt> ASSISTANT:"
```

For multiple turns conversation:


```
"USER: <image>\n<prompt1> ASSISTANT: <answer1></s>USER: <prompt2> ASSISTANT: <answer2></s>USER: <prompt3> ASSISTANT:"
```

🚀 **Bonus:** If you’re using `transformers>=4.49.0`, you can also get a vectorized output from `apply_chat_template`. See the **Usage Examples** below for more details on how to use it.

## Usage examples

### Single input inference


```
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```

### Batched inference

LLaVa also supports batched inference. Here is how you can do it:


```
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


# Prepare a batch of two prompts
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
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
    return_tensors="pt"
).to(model.device, torch.float16)


# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```

## Note regarding reproducing original implementation

In order to match the logits of the [original implementation](https://github.com/haotian-liu/LLaVA/tree/main), one needs to additionally specify `do_pad=True` when instantiating `LlavaImageProcessor`:


```
from transformers import LlavaImageProcessor

image_processor = LlavaImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", do_pad=True)
```

### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization, please refer to the [Flash Attention 2 section of performance docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BEiT.

​
 Image-to-Text

* A [Google Colab demo](https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing) on how to run Llava on a free-tier Google colab instance leveraging 4-bit inference.
* A [similar notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Inference_with_LLaVa_for_multimodal_generation.ipynb) showcasing batched inference. 🌎

## LlavaConfig

### class transformers.LlavaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/configuration_llava.py#L24)

( vision\_config = None text\_config = None image\_token\_index = 32000 projector\_hidden\_act = 'gelu' vision\_feature\_select\_strategy = 'default' vision\_feature\_layer = -2 image\_seq\_length = 576 multimodal\_projector\_bias = True \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `CLIPVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`) —
  The config object or dictionary of the text backbone.
* **image\_token\_index** (`int`, *optional*, defaults to 32000) —
  The image token index to encode the image prompt.
* **projector\_hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The activation function used by the multimodal projector.
* **vision\_feature\_select\_strategy** (`str`, *optional*, defaults to `"default"`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*, defaults to -2) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **image\_seq\_length** (`int`, *optional*, defaults to 576) —
  Sequence length of one image embedding.
* **multimodal\_projector\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the multimodal projector.

This is the configuration class to store the configuration of a [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration). It is used to instantiate an
Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Llava-9B.

e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

>>> # Initializing a CLIP-vision config
>>> vision_config = CLIPVisionConfig()

>>> # Initializing a Llama config
>>> text_config = LlamaConfig()

>>> # Initializing a Llava llava-1.5-7b style configuration
>>> configuration = LlavaConfig(vision_config, text_config)

>>> # Initializing a model from the llava-1.5-7b style configuration
>>> model = LlavaForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LlavaImageProcessor

### class transformers.LlavaImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/image_processing_llava.py#L53)

( do\_pad: bool = False do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True \*\*kwargs  )

Parameters

* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Whether to pad the image to a square based on the longest edge.
  The padding value is determined by the `image_mean` parameter.
  Can be overridden by `do_pad` in the `preprocess` method.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by
  `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 224}`):
  Size of the image after resizing. The shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
  method.
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

Constructs a LLaVa image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/image_processing_llava.py#L279)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_pad: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: typing.Optional[PIL.Image.Resampling] = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[int] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the image to a square based on the longest edge.
  The padding value is determined by the `image_mean` parameter.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. Shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
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

Preprocess an image or batch of images.

## LlavaImageProcessorFast

### class transformers.LlavaImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/image_processing_llava_fast.py#L70)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.llava.image\_processing\_llava\_fast.LlavaFastImageProcessorKwargs]  )

Constructs a fast Llava image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/image_processing_llava_fast.py#L88)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.llava.image\_processing\_llava\_fast.LlavaFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the image to a square based on the longest edge.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## LlavaProcessor

### class transformers.LlavaProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/processing_llava.py#L45)

( image\_processor = None tokenizer = None patch\_size = None vision\_feature\_select\_strategy = None chat\_template = None image\_token = '<image>' num\_additional\_image\_tokens = 0 \*\*kwargs  )

Parameters

* **image\_processor** ([LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **patch\_size** (`int`, *optional*) —
  Patch size from the vision tower.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Should be same as in model’s config
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.
* **image\_token** (`str`, *optional*, defaults to `"<image>"`) —
  Special token used to denote image location.
* **num\_additional\_image\_tokens** (`int`, *optional*, defaults to 0) —
  Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
  extra tokens appended, no need to set this arg.

Constructs a LLaVa processor which wraps a LLaVa image processor and a LLaMa tokenizer into a single processor.

[LlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaProcessor) offers all the functionalities of [LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## LlavaModel

### class transformers.LlavaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/modeling_llava.py#L134)

( config: LlavaConfig  )

Parameters

* **config** ([LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Llava model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/modeling_llava.py#L245)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None image\_sizes: Tensor = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.llava.modeling_llava.LlavaModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor). See [LlavaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaProcessor) uses
  [LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor) for processing images).
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
* **image\_sizes** (`torch.Tensor` of shape `(batch_size, 2)`) —
  The sizes of the images in the batch, being (height, width) for each image.

Returns

`transformers.models.llava.modeling_llava.LlavaModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llava.modeling_llava.LlavaModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig)) and inputs.

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

The [LlavaModel](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/modeling_llava.py#L157)

( pixel\_values: FloatTensor vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None \*\*kwargs  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) —
  The tensors corresponding to the input images.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_images, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/modeling_llava.py#L221)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## LlavaForConditionalGeneration

### class transformers.LlavaForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/modeling_llava.py#L325)

( config: LlavaConfig  )

Parameters

* **config** ([LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LLAVA model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llava/modeling_llava.py#L382)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 image\_sizes: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.llava.modeling_llava.LlavaCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor). See [LlavaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaProcessor) uses
  [LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor) for processing images).
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
* **image\_sizes** (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) —
  The sizes of the images in the batch, being (height, width) for each image.

Returns

`transformers.models.llava.modeling_llava.LlavaCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llava.modeling_llava.LlavaCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig)) and inputs.

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
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, LlavaForConditionalGeneration

>>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
>>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

>>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=15)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llava.md)
