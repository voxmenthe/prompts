*This model was released on 2024-09-27 and added to Hugging Face Transformers on 2025-01-10.*

# Emu3

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Emu3 model was proposed in [Emu3: Next-Token Prediction is All You Need](https://huggingface.co/papers/2409.18869) by Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, Zhongyuan Wang.

Emu3 is a multimodal LLM that uses vector quantization to tokenize images into discrete tokens. Discretized image tokens are later fused with text token ids for image and text generation. The model can additionally generate images by predicting image token ids.

The abstract from the paper is the following:

*While next-token prediction is considered a promising path towards artificial general intelligence, it has struggled to excel in multimodal tasks, which are still dominated by diffusion models (e.g., Stable Diffusion) and compositional approaches (e.g., CLIP combined with LLMs). In this paper, we introduce Emu3, a new suite of state-of-the-art multimodal models trained solely with next-token prediction. By tokenizing images, text, and videos into a discrete space, we train a single transformer from scratch on a mixture of multimodal sequences. Emu3 outperforms several well-established task-specific models in both generation and perception tasks, surpassing flagship models such as SDXL and LLaVA-1.6, while eliminating the need for diffusion or compositional architectures. Emu3 is also capable of generating high-fidelity video via predicting the next token in a video sequence. We simplify complex multimodal model designs by converging on a singular focus: tokens, unlocking great potential for scaling both during training and inference. Our results demonstrate that next-token prediction is a promising path towards building general multimodal intelligence beyond language. We open-source key techniques and models to support further research in this direction.*

Tips:

* We advise users to set `processor.tokenizer.padding_side = "left"` before batched generation as it leads to more accurate results.
* Note that the model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.
* Emu3 has two different checkpoints for image-generation and text-generation, make sure to use the correct checkpoint when loading the model. To generate an image, it is advised to use `prefix_constraints` so that the generated tokens are sampled only from possible image tokens. See more below for usage examples.

Emu3 implementation in Transformers uses a special image token to indicate where to merge image embeddings. The special image token isn’t new and uses one of the reserved tokens: `<|extra_0|>`. You have to add `<image>` to your prompt in the place where the image should be embedded for correct generation.

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/baaivision/Emu3).

## Usage example

### Text generation inference

Here’s how to load the model and perform inference in half-precision (`torch.bfloat16`) to generate textual output from text or text and image inputs:


```
from transformers import Emu3Processor, Emu3ForConditionalGeneration
import torch
from PIL import Image
import requests

processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")
model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", dtype=torch.bfloat16, device_map="auto")

# prepare image and text prompt
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Image generation inference

Emu3 can also generate images from textual input. Here is how you can do it:


```
processor = Emu3Processor.from_pretrained("BAAI/Emu3-Gen-hf")
model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Gen-hf", dtype="bfloat16", device_map="auto", attn_implementation="flash_attention_2")


inputs = processor(
    text=["a portrait of young girl. masterpiece, film grained, best quality.", "a dog running under the rain"],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
)
inputs = inputs.to(device=model.device, dtype=torch.bfloat16)

neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
neg_inputs = processor(text=[neg_prompt] * 2, return_tensors="pt").to(device=model.device)

image_sizes = inputs.pop("image_sizes")
HEIGHT, WIDTH = image_sizes[0]
VISUAL_TOKENS = model.vocabulary_mapping.image_tokens

def prefix_allowed_tokens_fn(batch_id, input_ids):
    height, width = HEIGHT, WIDTH
    visual_tokens = VISUAL_TOKENS
    image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
    pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position
    if offset % (width + 1) == 0:
        return (eol_token_id, )
    elif offset == (width + 1) * height + 1:
        return (eof_token_id, )
    elif offset == (width + 1) * height + 2:
        return (eoi_token_id, )
    elif offset == (width + 1) * height + 3:
        return (eos_token_id, )
    elif offset > (width + 1) * height + 3:
        return (pad_token_id, )
    else:
        return visual_tokens


out = model.generate(
    **inputs,
    max_new_tokens=50_000, # make sure to have enough tokens for one image
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    return_dict_in_generate=True,
    negative_prompt_ids=neg_inputs.input_ids, # indicate for Classifier-Free Guidance
    negative_prompt_attention_mask=neg_inputs.attention_mask,
)

image = model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1]: ], height=HEIGHT, width=WIDTH)
images = processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image") # internally we convert to np but it's not supported in bf16 precision
for i, image in enumerate(images['pixel_values']):
    image.save(f"result{i}.png")
```

## Emu3Config

### class transformers.Emu3Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/configuration_emu3.py#L279)

( vq\_config: typing.Union[dict, transformers.models.emu3.configuration\_emu3.Emu3VQVAEConfig] = None text\_config: typing.Union[dict, transformers.models.emu3.configuration\_emu3.Emu3TextConfig] = None vocabulary\_map: typing.Optional[dict[int, int]] = None \*\*kwargs  )

Parameters

* **vq\_config** (`Union[Dict, Emu3VQVAEConfig]`, *optional*) —
  Emu3VQVAEConfig instance containing the configuration for the VQ-VAE model.
* **text\_config** (`Union[Dict, Emu3TextConfig]“, *optional*) —
  Emu3TextConfig instance containing the configuration for the language model.
* **vocabulary\_map** (`dict`, *optional*) —
  A dictionary containing the vocabulary map from the tokenizer. Used to obtain tokens from the image inputs.

This is the configuration class to store the configuration of a [Emu3Model](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Model). It is used to instantiate a
emu3 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
[Emu3-community/Emu3-Chat-hf](https://huggingface.co/Emu3-community/Emu3-Chat-hf).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Emu3VQVAEConfig

### class transformers.Emu3VQVAEConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/configuration_emu3.py#L23)

( codebook\_size: int = 32768 embed\_dim: int = 4 latent\_channels: int = 4 double\_latent: bool = False in\_channels: int = 3 out\_channels: int = 3 temporal\_downsample\_factor: int = 4 base\_channels: int = 256 channel\_multiplier: list = [1, 2, 2, 4] num\_res\_blocks: int = 2 attn\_resolutions: list = [3] hidden\_size: int = 1024 num\_attention\_heads: int = 1 attention\_dropout: float = 0.0 \*\*kwargs  )

Parameters

* **codebook\_size** (`int`, *optional*, defaults to 32768) —
  Codebook size of the VQ model.
* **embed\_dim** (`int`, *optional*, defaults to 4) —
  Dimension of the quantized vector in codebook.
* **latent\_channels** (`int`, *optional*, defaults to 4) —
  Dimension of the output channel of encoder and the input channel of decoder
* **double\_latent** (`bool`, *optional*, defaults to `False`) —
  Whether double the output dim of the encoder.
* **in\_channels** (`int`, *optional*, defaults to 3) —
  Input channel of encoder.
* **out\_channels** (`int`, *optional*, defaults to 3) —
  Output channel of decoder.
* **temporal\_downsample\_factor** (`int`, *optional*, defaults to 4) —
  Temporal downsample factor.
* **base\_channels** (`int`, *optional*, defaults to 256) —
  Basic channel number of the intermediate blocks.
* **channel\_multiplier** (`list[int]`, *optional*, defaults to `[1, 2, 2, 4]`) —
  Channel scaling factor of the intermediate blocks.
* **num\_res\_blocks** (`int`, *optional*, defaults to 2) —
  Residual block number in each stage.
* **attn\_resolutions** (`list[int]`, *optional*, defaults to `[3]`) —
  Stage indices to apply attention.
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimension of the hidden representations in the attention layer.
* **num\_attention\_heads** (`int`, *optional*, defaults to 1) —
  Number of attention heads for each attention layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.

This is the configuration class to store the configuration of a [Emu3VQVAE](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3VQVAE). It is used to instantiate an VQ-VAE
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a configuration to the VQ model presented in Emu3 paper.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Emu3VQVAE, Emu3VQVAEConfig

>>> # Initializing a video VQ model of Emu3 configuration
>>> configuration = Emu3VQVAEConfig()

>>> # Initializing a model from the Emu3 VQ model style configuration
>>> model = Emu3VQVAE(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Emu3TextConfig

### class transformers.Emu3TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/configuration_emu3.py#L113)

( vocab\_size: int = 184622 hidden\_size: int = 4096 intermediate\_size: int = 14336 num\_hidden\_layers: int = 32 num\_attention\_heads: int = 32 num\_key\_value\_heads: typing.Optional[int] = 8 hidden\_act: str = 'silu' max\_position\_embeddings: int = 9216 rms\_norm\_eps: float = 1e-05 use\_cache: bool = True pad\_token\_id: int = 151643 bos\_token\_id: int = 151849 eos\_token\_id: int = 151850 tie\_word\_embeddings: bool = False rope\_theta: float = 1000000.0 rope\_scaling: typing.Optional = None mlp\_bias = False attention\_bias = False attention\_dropout: float = 0.1 initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 184622) —
  Vocabulary size of the Emu3 model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Emu3Model](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Model)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 14336) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to` num\_attention\_heads`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 9216) —
  The maximum sequence length that this model might ever be used with. Emu supports up to 9216 tokens,
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 151643) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 151849) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 151850) —
  End of stream token id.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **rope\_theta** (`float`, *optional*, defaults to 1000000.0) —
  The base period of the RoPE embeddings.
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
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in up\_proj, down\_proj and gate\_proj layers in the MLP layers.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [Emu3TextModel](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3TextModel). It is used to instantiate a
emu3 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
[Emu3-community/Emu3-Chat-hf](https://huggingface.co/Emu3-community/Emu3-Chat-hf).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Emu3Model, Emu3Config

>>> # Initializing a Emu3-community/Emu3-Chat-hf style configuration
>>> configuration = Emu3Config()

>>> # Initializing a model from the Emu3-community/Emu3-Chat-hf style configuration
>>> model = Emu3Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Emu3Processor

### class transformers.Emu3Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/processing_emu3.py#L56)

( image\_processor tokenizer chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([Emu3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ImageProcessor)) —
  The image processor is a required input.
* **tokenizer** (`Emu3TokenizerFast`) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a Emu3 processor which wraps a Emu3 image processor and a GPT2 tokenizer into a single
processor.

[Emu3Processor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Processor) offers all the functionalities of [Emu3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ImageProcessor) and [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast).
See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Emu3ImageProcessor

### class transformers.Emu3ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/image_processing_emu3.py#L101)

( do\_resize: bool = True resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True do\_pad: bool = True min\_pixels: int = 262144 max\_pixels: int = 1048576 spatial\_factor: int = 8 \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions.
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
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
* **min\_pixels** (`int`, *optional*, defaults to `512 * 512`) —
  The min pixels of the image to resize the image.
* **max\_pixels** (`int`, *optional*, defaults to `1024 * 1024`) —
  The max pixels of the image to resize the image.
* **spatial\_factor** (`int`, *optional*, defaults to 8) —
  The spatial downsample factor the image will be downsampled in feature extracting phase

Constructs a Emu3 image processor that dynamically resizes images based on the original images.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/image_processing_emu3.py#L306)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None do\_pad: bool = True return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
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

## Emu3VQVAE

### class transformers.Emu3VQVAE

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L927)

( config: Emu3VQVAEConfig  )

Parameters

* **config** ([Emu3VQVAEConfig](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3VQVAEConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The VQ-VAE model used in Emu3 for encoding/decoding images into discrete tokens.
This model follows the “Make-a-scene: Scene-based text-to-image generation with human priors” paper from
[Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv
Taigman](https://huggingface.co/papers/2203.13131).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### \_forward\_unimplemented

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/torch/nn/modules/module.py#L388)

( \*input: typing.Any  )

Define the computation performed at every call.

Should be overridden by all subclasses.

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## Emu3TextModel

### class transformers.Emu3TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1147)

( config: Emu3Config  )

Parameters

* **config** ([Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Emu3 Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1169)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config)) and inputs.

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

The [Emu3TextModel](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Emu3Model

### class transformers.Emu3Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1307)

( config  )

#### decode\_image\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1366)

( image\_tokens: LongTensor height: int width: int  )

Parameters

* **image\_tokens** (`torch.LongTensor` of shape `(batch_size, num_of_tokens)`) —
  The tensors corresponding to the input images.
* **height** (`int`) —
  Height of the generated image before upsampling.
* **width** (`int`) —
  Width of the generated image before upsampling.

Decodes generated image tokens from language model to continuous pixel values
with VQGAN module via upsampling.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1409)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None image\_sizes: Tensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Emu3Processor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Processor) uses
  `image_processor_class` for processing images).
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  The sizes of the images in the batch, being (height, width) for each image. Image sizes can be obtained using
  [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [Emu3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([]`Emu3Processor`] uses
  [Emu3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ImageProcessor) for processing images).
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Emu3Model](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1348)

( pixel\_values: FloatTensor image\_sizes: LongTensor  )

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch\_size, num\_channels, image\_size, image\_size)) —
  The tensors corresponding to the input images.

Tokenizes images into discrete tokens with VQGAN module and embeds
them with text embeddings layer

#### get\_image\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1331)

( pixel\_values: FloatTensor image\_sizes: LongTensor  )

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images.
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  The sizes of the images in the batch, being (height, width) for each image.

Tokenizes images into discrete tokens with VQGAN module. Converts
obtained image tokens into BPE tokens and wraps with “boi” and “eoi”
special tokens.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1385)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## Emu3ForCausalLM

### class transformers.Emu3ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1231)

( config  )

Parameters

* **config** ([Emu3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Emu3 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1246)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
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

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Emu3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import Emu3Processor, Emu3ForConditionalGeneration
>>> import torch
>>> import requests
>>> from PIL import Image

>>> model = Emu3ForCausalLM.from_pretrained("BAAI/Emu3-Chat-hf", dtype=torch.bfloat16)
>>> processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

>>> inputs = processor(text=["Can you write me a poem about winter."], return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
>>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Emu3ForConditionalGeneration

### class transformers.Emu3ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1460)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/emu3/modeling_emu3.py#L1507)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None image\_sizes: Tensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Emu3Processor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Processor) uses
  `image_processor_class` for processing images).
* **image\_sizes** (`torch.LongTensor` of shape `(batch_size, 2)`) —
  The sizes of the images in the batch, being (height, width) for each image. Image sizes can be obtained using
  [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See [Emu3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([]`Emu3Processor`] uses
  [Emu3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ImageProcessor) for processing images).
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Emu3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import Emu3Processor, Emu3ForConditionalGeneration
>>> import torch
>>> import requests
>>> from PIL import Image

>>> model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", dtype=torch.bfloat16)
>>> processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

>>> conversation = [
...     {
...     "role": "system",
...     "content": [
...         {"type": "text", "text": "You are a helpful assistant."},
...         ],
...     },
...     {
...     "role": "user",
...     "content": [
...         {"type": "image"},
...         {"type": "text", "text": "Please describe the image."},
...         ],
...     },
... ]

>>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
>>> image = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)

>>> inputs = processor(images=[image], text=[prompt], return_tensors="pt").to(model.device, torch.bfloat16)

>>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
>>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/emu3.md)
