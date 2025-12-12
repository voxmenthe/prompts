*This model was released on 2024-09-03 and added to Hugging Face Transformers on 2025-01-31.*

# GOT-OCR2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The GOT-OCR2 model was proposed in [General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model](https://huggingface.co/papers/2409.01704) by Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, Xiangyu Zhang.

The abstract from the paper is the following:

*Traditional OCR systems (OCR-1.0) are increasingly unable to meet people’snusage due to the growing demand for intelligent processing of man-made opticalncharacters. In this paper, we collectively refer to all artificial optical signals (e.g., plain texts, math/molecular formulas, tables, charts, sheet music, and even geometric shapes) as “characters” and propose the General OCR Theory along with an excellent model, namely GOT, to promote the arrival of OCR-2.0. The GOT, with 580M parameters, is a unified, elegant, and end-to-end model, consisting of a high-compression encoder and a long-contexts decoder. As an OCR-2.0 model, GOT can handle all the above “characters” under various OCR tasks. On the input side, the model supports commonly used scene- and document-style images in slice and whole-page styles. On the output side, GOT can generate plain or formatted results (markdown/tikz/smiles/kern) via an easy prompt. Besides, the model enjoys interactive OCR features, i.e., region-level recognition guided by coordinates or colors. Furthermore, we also adapt dynamic resolution and multipage OCR technologies to GOT for better practicality. In experiments, we provide sufficient results to prove the superiority of our model.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/got_ocr_overview.png) GOT-OCR2 training stages. Taken from the [original paper.](https://huggingface.co/papers/2409.01704)

Tips:

GOT-OCR2 works on a wide range of tasks, including plain document OCR, scene text OCR, formatted document OCR, and even OCR for tables, charts, mathematical formulas, geometric shapes, molecular formulas and sheet music. While this implementation of the model will only output plain text, the outputs can be further processed to render the desired format, with packages like `pdftex`, `mathpix`, `matplotlib`, `tikz`, `verovio` or `pyecharts`.
The model can also be used for interactive OCR, where the user can specify the region to be recognized by providing the coordinates or the color of the region’s bounding box.

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/Ucas-HaoranWei/GOT-OCR2.0).

## Usage example

### Plain text inference


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, infer_device

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
>>> inputs = processor(image, return_tensors="pt", device=device).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\nName/Phone Ext. : (...)"
```

### Plain text inference batched


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, infer_device

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
>>> image2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"

>>> inputs = processor([image1, image2], return_tensors="pt", device=device).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4,
... )

>>> processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
["Reducing the number", "R&D QUALITY"]
```

### Formatted text inference

GOT-OCR2 can also generate formatted text, such as markdown or LaTeX. Here is an example of how to generate formatted text:


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, infer_device

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/latex.png"
>>> inputs = processor(image, return_tensors="pt", format=True, device=device).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"\\author{\nHanwen Jiang*{@html "quad\\quadquad"} Arjun Karpur{@html "daggerquad{ }^{\\dagger} \\quaddaggerquad"} Bingyi Cao{@html "daggerquad{ }^{\\dagger} \\quaddaggerquad"} (...)"
```

### Inference on multiple pages

Although it might be reasonable in most cases to use a “for loop” for multi-page processing, some text data with formatting across several pages make it necessary to process all pages at once. GOT introduces a multi-page OCR (without “for loop”) feature, where multiple pages can be processed by the model at once, with the output being one continuous text.
Here is an example of how to process multiple pages at once:


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, infer_device

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/page1.png"
>>> image2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/page2.png"
>>> inputs = processor([image1, image2], return_tensors="pt", multi_page=True, format=True, device=device).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"\\title{\nGeneral OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model\n}\n\\author{\nHaoran Wei (...)"
```

### Inference on cropped patches

GOT supports a 1024×1024 input resolution, which is sufficient for most OCR tasks, such as scene OCR or processing A4-sized PDF pages. However, certain scenarios, like horizontally stitched two-page PDFs commonly found in academic papers or images with unusual aspect ratios, can lead to accuracy issues when processed as a single image. To address this, GOT can dynamically crop an image into patches, process them all at once, and merge the results for better accuracy with such inputs.
Here is an example of how to process cropped patches:


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, infer_device

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", dtype=torch.bfloat16, device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/one_column.png"
>>> inputs = processor(image, return_tensors="pt", format=True, crop_to_patches=True, max_patches=3, device=device).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"on developing architectural improvements to make learnable matching methods generalize.\nMotivated by the above observations, (...)"
```

### Inference on a specific region

GOT supports interactive OCR, where the user can specify the region to be recognized by providing the coordinates or the color of the region’s bounding box. Here is an example of how to process a specific region:


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
>>> inputs = processor(image, return_tensors="pt", color="green", device=device).to(device) # or box=[x1, y1, x2, y2] for coordinates (image pixels)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"You should keep in mind what features from the module should be used, especially \nwhen you’re planning to sell a template."
```

### Inference on general OCR data example: sheet music

Although this implementation of the model will only output plain text, the outputs can be further processed to render the desired format, with packages like `pdftex`, `mathpix`, `matplotlib`, `tikz`, `verovio` or `pyecharts`.
Here is an example of how to process sheet music:


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, infer_device
>>> import verovio

>>> device = infer_device()
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/sheet_music.png"
>>> inputs = processor(image, return_tensors="pt", format=True, device=device).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> outputs = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
>>> tk = verovio.toolkit()
>>> tk.loadData(outputs)
>>> tk.setOptions(
...     {
...         "pageWidth": 2100,
...         "pageHeight": 800,
...         "footer": "none",
...         "barLineWidth": 0.5,
...         "beamMaxSlope": 15,
...         "staffLineWidth": 0.2,
...         "spacingStaff": 6,
...     }
... )
>>> tk.getPageCount()
>>> svg = tk.renderToSVG()
>>> svg = svg.replace('overflow="inherit"', 'overflow="visible"')
>>> with open("output.svg", "w") as f:
>>>     f.write(svg)
```

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sheet_music.svg)

## GotOcr2Config

### class transformers.GotOcr2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/configuration_got_ocr2.py#L118)

( vision\_config = None text\_config = None image\_token\_index = 151859 image\_seq\_length = 576 pad\_token\_id = -1 \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `CLIPVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`) —
  The config object or dictionary of the text backbone.
* **image\_token\_index** (`int`, *optional*, defaults to 151859) —
  The image token index to encode the image prompt.
* **image\_seq\_length** (`int`, *optional*, defaults to 576) —
  Sequence length of one image embedding.
* **pad\_token\_id** (`int`, *optional*, defaults to -1) —
  Padding token id.

This is the configuration class to store the configuration of a [GotOcr2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ForConditionalGeneration). It is used to instantiate a
GotOcr2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of GOT-OCR-2.0.

e.g [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import GotOcr2ForConditionalGeneration, GotOcr2Config

>>> # Initializing a GotOcr2 style configuration
>>> configuration = GotOcr2Config()

>>> # Initializing a model from the Qwen2-VL-7B style configuration
>>> model = GotOcr2ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GotOcr2VisionConfig

### class transformers.GotOcr2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/configuration_got_ocr2.py#L27)

( hidden\_size = 768 output\_channels = 256 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 image\_size = 1024 patch\_size = 16 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 initializer\_range = 1e-10 qkv\_bias = True use\_abs\_pos = True use\_rel\_pos = True window\_size = 14 global\_attn\_indexes = [2, 5, 8, 11] mlp\_dim = 3072 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **output\_channels** (`int`, *optional*, defaults to 256) —
  Dimensionality of the output channels in the Patch Encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input image.
* **image\_size** (`int`, *optional*, defaults to 1024) —
  Expected resolution. Target size of the resized input image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  Size of the patches to be extracted from the input image.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string)
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 1e-10) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to query, key, value projections.
* **use\_abs\_pos** (`bool`, *optional*, defaults to `True`) —
  Whether to use absolute position embedding.
* **use\_rel\_pos** (`bool`, *optional*, defaults to `True`) —
  Whether to use relative position embedding.
* **window\_size** (`int`, *optional*, defaults to 14) —
  Window size for relative position.
* **global\_attn\_indexes** (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`) —
  The indexes of the global attention layers.
* **mlp\_dim** (`int`, *optional*, defaults to 3072) —
  The dimensionality of the MLP layer in the Transformer encoder.

This is the configuration class to store the configuration of a `GotOcr2VisionModel`. It is used to instantiate a GOT\_OCR2
vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
defaults will yield a similar configuration to that of the SAM ViT-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## GotOcr2ImageProcessor

### class transformers.GotOcr2ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2.py#L126)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None crop\_to\_patches: bool = False min\_patches: int = 1 max\_patches: int = 12 resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"height" -- 384, "width": 384}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **crop\_to\_patches** (`bool`, *optional*, defaults to `False`) —
  Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
  `preprocess` method.
* **min\_patches** (`int`, *optional*, defaults to 1) —
  The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
* **max\_patches** (`int`, *optional*, defaults to 12) —
  The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a GOT\_OCR2 image processor.

#### crop\_image\_to\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2.py#L418)

( images: ndarray min\_patches: int max\_patches: int use\_thumbnail: bool = True patch\_size: typing.Union[tuple, int, dict, NoneType] = None data\_format: ChannelDimension = None  ) → list`PIL.Image.Image` or list[np.ndarray]

Parameters

* **images** (`np.ndarray`) —
  The image to be cropped.
* **min\_patches** (`int`) —
  The minimum number of patches to be extracted from the image.
* **max\_patches** (`int`) —
  The maximum number of patches to be extracted from the image.
* **use\_thumbnail** (`bool`, *optional*, defaults to `True`) —
  Whether to add a thumbnail image to the list of cropped patches.
* **patch\_size** (`int`, `tuple[int, int]`, `dict`, *optional*) —
  The size of the output patches.
* **data\_format** (`ChannelDimension`, *optional*) —
  The format of the image data. If `None`, the format is inferred from the input image.

Returns

list`PIL.Image.Image` or list[np.ndarray]

The list of cropped images.

Crop the image to patches and return a list of cropped images.
The number of patches and their grid arrangement are determined by the original image size,
the target patch size and the minimum and maximum number of patches.
The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

#### get\_number\_of\_image\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2.py#L496)

( height: int width: int images\_kwargs = None  ) → `int`

Parameters

* **height** (`int`) —
  Height of the input image.
* **width** (`int`) —
  Width of the input image.
* **images\_kwargs** (`dict`, *optional*) —
  Any kwargs to override defaults of the image processor.

Returns

`int`

Number of patches per image.

A utility that returns number patches for a given image size.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2.py#L253)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None crop\_to\_patches: typing.Optional[bool] = None min\_patches: typing.Optional[int] = None max\_patches: typing.Optional[int] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Controls the size of the image after `resize`. The shortest edge of the image is resized to
  `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
  is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
  edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
* **crop\_to\_patches** (`bool`, *optional*, defaults to `self.crop_to_patches`) —
  Whether to crop the image to patches.
* **min\_patches** (`int`, *optional*, defaults to `self.min_patches`) —
  The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`.
* **max\_patches** (`int`, *optional*, defaults to `self.max_patches`) —
  The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to normalize the image by if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
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

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2.py#L205)

( image: ndarray size: dict resample: Resampling = <Resampling.BICUBIC: 3> data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
* **data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Returns

`np.ndarray`

The resized image.

Resize an image to `(size["height"], size["width"])`.

## GotOcr2ImageProcessorFast

### class transformers.GotOcr2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2_fast.py#L67)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.got\_ocr2.image\_processing\_got\_ocr2\_fast.GotOcr2FastImageProcessorKwargs]  )

Constructs a fast Got Ocr2 image processor.

#### crop\_image\_to\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2_fast.py#L88)

( images: torch.Tensor min\_patches: int max\_patches: int use\_thumbnail: bool = True patch\_size: typing.Union[tuple, int, dict, NoneType] = None interpolation: typing.Optional[ForwardRef('F.InterpolationMode')] = None  ) → list`PIL.Image.Image` or list[np.ndarray]

Parameters

* **images** (`torch.Tensor`) —
  The images to be cropped.
* **min\_patches** (`int`) —
  The minimum number of patches to be extracted from the image.
* **max\_patches** (`int`) —
  The maximum number of patches to be extracted from the image.
* **use\_thumbnail** (`bool`, *optional*, defaults to `True`) —
  Whether to add a thumbnail image to the list of cropped patches.
* **patch\_size** (`int`, `tuple[int, int]`, `dict`, *optional*) —
  The size of the output patches.
  The format of the image data. If `None`, the format is inferred from the input image.

Returns

list`PIL.Image.Image` or list[np.ndarray]

The list of cropped images.

Crop the images to patches and return a list of cropped images.
The number of patches and their grid arrangement are determined by the original image size,
the target patch size and the minimum and maximum number of patches.
The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

#### get\_number\_of\_image\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2_fast.py#L226)

( height: int width: int images\_kwargs = None  ) → `int`

Parameters

* **height** (`int`) —
  Height of the input image.
* **width** (`int`) —
  Width of the input image.
* **images\_kwargs** (`dict`, *optional*) —
  Any kwargs to override defaults of the image processor.

Returns

`int`

Number of patches per image.

A utility that returns number patches for a given image size.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/image_processing_got_ocr2_fast.py#L84)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.got\_ocr2.image\_processing\_got\_ocr2\_fast.GotOcr2FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **crop\_to\_patches** (`bool`, *optional*, defaults to `False`) —
  Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
  `preprocess` method.
* **min\_patches** (`int`, *optional*, defaults to 1) —
  The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
* **max\_patches** (`int`, *optional*, defaults to 12) —
  The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## GotOcr2Processor

### class transformers.GotOcr2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/processing_got_ocr2.py#L83)

( image\_processor = None tokenizer = None chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a GotOcr2 processor which wraps a [GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) and
`PretrainedTokenizerFast` tokenizer into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## GotOcr2Model

### class transformers.GotOcr2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L534)

( config: GotOcr2Config  )

Parameters

* **config** ([GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GotOcr2 model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L596)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.got_ocr2.modeling_got_ocr2.GotOcr2ModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor). See [GotOcr2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([GotOcr2Processor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Processor) uses
  [GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) for processing images).
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

`transformers.models.got_ocr2.modeling_got_ocr2.GotOcr2ModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.got_ocr2.modeling_got_ocr2.GotOcr2ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config)) and inputs.

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

The [GotOcr2Model](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L557)

( pixel\_values: FloatTensor  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) —

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_images, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L572)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## GotOcr2ForConditionalGeneration

### class transformers.GotOcr2ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L660)

( config: GotOcr2Config  )

Parameters

* **config** ([GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GOT\_OCR2 model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L717)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.got_ocr2.modeling_got_ocr2.GotOcr2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor). See [GotOcr2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([GotOcr2Processor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Processor) uses
  [GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) for processing images).
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

`transformers.models.got_ocr2.modeling_got_ocr2.GotOcr2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.got_ocr2.modeling_got_ocr2.GotOcr2CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config)) and inputs.

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

The [GotOcr2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GotOcr2ForConditionalGeneration, TextStreamer

>>> model = GotOcr2ForConditionalGeneration.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf").to("cuda")
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(image, return_tensors="pt", color="green").to("cuda")

>>> # Generate
>>> streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer = processor.tokenizer,
...     stop_strings='<|im_end|>',
...     streamer=streamer,
...     max_new_tokens=4096,
... )
"You should keep in mind what features from the module should be used, especially
when you're planning to sell a template."
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/got_ocr2.md)
