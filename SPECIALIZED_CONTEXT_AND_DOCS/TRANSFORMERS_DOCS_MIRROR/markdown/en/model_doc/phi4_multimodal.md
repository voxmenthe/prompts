*This model was released on 2025-03-03 and added to Hugging Face Transformers on 2025-03-25.*

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)

## Phi4 Multimodal

[Phi4 Multimodal](https://huggingface.co/papers/2503.01743) is a multimodal model capable of text, image, and speech and audio inputs or any combination of these. It features a mixture of LoRA adapters for handling different inputs, and each input is routed to the appropriate encoder.

You can find all the original Phi4 Multimodal checkpoints under the [Phi4](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4) collection.

This model was contributed by [cyrilvallez](https://huggingface.co/cyrilvallez).

Click on the Phi-4 Multimodal in the right sidebar for more examples of how to apply Phi-4 Multimodal to different tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
from transformers import pipeline
generator = pipeline("text-generation", model="microsoft/Phi-4-multimodal-instruct", dtype="auto", device=0)

prompt = "Explain the concept of multimodal AI in simple terms."

result = generator(prompt, max_length=50)
print(result[0]['generated_text'])
```

## Notes

The example below demonstrates inference with an audio and text input.


```
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, infer_device

model_path = "microsoft/Phi-4-multimodal-instruct"
device = f"{infer_device()}:0"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device,  dtype=torch.float16)

model.load_adapter(model_path, adapter_name="speech", device_map=device, adapter_kwargs={"subfolder": 'speech-lora'})
model.set_adapter("speech")
audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
            {"type": "text", "text": "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the origina transcript and the translation."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
```

## Phi4MultimodalFeatureExtractor

### class transformers.Phi4MultimodalFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/feature_extraction_phi4_multimodal.py#L36)

( feature\_size: int = 80 sampling\_rate: int = 16000 hop\_length: int = 160 n\_fft: int = 512 win\_length: int = 400 preemphasis: float = 0.97 padding\_value: float = 0.0 audio\_compression\_rate: int = 8 audio\_downsample\_rate: int = 1 audio\_feat\_stride: int = 1 mel\_min\_frequency: float = 0 mel\_max\_frequency: float = 7690 \*\*kwargs  )

## Phi4MultimodalImageProcessorFast

### class transformers.Phi4MultimodalImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/image_processing_phi4_multimodal_fast.py#L61)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.phi4\_multimodal.image\_processing\_phi4\_multimodal\_fast.Phi4MultimodalFastImageProcessorKwargs]  )

Constructs a fast Phi4 Multimodal image processor.

#### pad\_to\_max\_num\_crops

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/image_processing_phi4_multimodal_fast.py#L150)

( images max\_crops = 5  )

images: B x 3 x H x W, B<=max\_crops

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/image_processing_phi4_multimodal_fast.py#L167)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.phi4\_multimodal.image\_processing\_phi4\_multimodal\_fast.Phi4MultimodalFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **patch\_size** (`int`, *optional*) —
  The size of the patch.
* **dynamic\_hd** (`int`, *optional*) —
  The maximum number of crops per image.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Phi4MultimodalProcessor

### class transformers.Phi4MultimodalProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/processing_phi4_multimodal.py#L41)

( image\_processor audio\_processor tokenizer \*\*kwargs  )

Parameters

* **image\_processor** (`Phi4MultimodalImageProcessorFast`) —
  The image processor to use for images.
* **audio\_processor** (`Phi4MultimodalFeatureExtractor`) —
  The audio processor to use for audio inputs.
* **tokenizer** (`GPT2TokenizerFast`) —
  The tokenizer to use for text.
* **fake\_image\_token\_pattern** (`str`, *optional*, defaults to `r"<\|image_\d+\|>"`) —
  The fake image token pattern.
* **fake\_audio\_token\_pattern** (`str`, *optional*, defaults to `r"<\|audio_\d+\|>"`) —
  The fake audio token pattern.

Constructs a Phi4Multimodal processor which raps an image processor, a audio processor, and a GPT tokenizer into a single processor.

[Phi4MultimodalProcessor](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalProcessor) offers all the functionalities of [Phi4MultimodalImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalImageProcessorFast) and [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Phi4MultimodalAudioConfig

### class transformers.Phi4MultimodalAudioConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L111)

( hidden\_size: int = 1024 intermediate\_size: int = 1536 num\_blocks: int = 24 num\_attention\_heads: int = 16 activation: str = 'swish' chunk\_size: int = -1 left\_chunk: int = 18 dropout\_rate: float = 0.0 ext\_pw\_out\_channel: int = 1024 depthwise\_seperable\_out\_channel: int = 1024 depthwise\_multiplier: int = 1 kernel\_size: int = 3 conv\_activation: str = 'swish' input\_size: int = 80 conv\_glu\_type: str = 'swish' time\_reduction: int = 8 bias\_max\_distance: int = 1000 bias\_symmetric: bool = False nemo\_activation: str = 'relu' nemo\_conv\_channels: int = 1024 downsample\_rate: int = 1 initializer\_range: float = 0.02 audio\_token\_id: int = 200011 feature\_layer: int = -2 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers.
* **intermediate\_size** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_blocks** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **activation** (`str`, *optional*, defaults to `"swish"`) —
  The non-linear activation function in the MLPs.
* **chunk\_size** (`int`, *optional*, defaults to -1) —
  The chunk size to create the masks.
* **left\_chunk** (`int`, *optional*, defaults to 18) —
  The left chunk to create the masks.
* **dropout\_rate** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio.
* **ext\_pw\_out\_channel** (`int`, *optional*, defaults to 1024) —
  Number of out channels in the point-wise conv modules.
* **depthwise\_seperable\_out\_channel** (`int`, *optional*, defaults to 1024) —
  Number of out channels in the depth-wise separable conv modules.
* **depthwise\_multiplier** (`int`, *optional*, defaults to 1) —
  Input size multiplier for the depth-wise separable conv modules.
* **kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size for the depth-wise separable conv modules.
* **conv\_activation** (`str`, *optional*, defaults to `"swish"`) —
  The non-linear activation function in the conv modules.
* **input\_size** (`int`, *optional*, defaults to 80) —
  Input size for the audio model.
* **conv\_glu\_type** (`str`, *optional*, defaults to `"swish"`) —
  The non-linear activation function in the point-wise conv modules.
* **time\_reduction** (`int`, *optional*, defaults to 8) —
  Time reduction (subsampling factor).
* **bias\_max\_distance** (`int`, *optional*, defaults to 1000) —
  Max distance for the relative attention bias module.
* **bias\_symmetric** (`bool`, *optional*, defaults to `False`) —
  Whether the relative attention bias should be symmetric or not.
* **nemo\_activation** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function in the nemo conv modules.
* **nemo\_conv\_channels** (`int`, *optional*, defaults to 1024) —
  Number of channels in the nemo conv modules.
* **downsample\_rate** (`int`, *optional*, defaults to 1) —
  Downsample rate for the audio feature extractor.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **audio\_token\_id** (`int`, *optional*, defaults to 200011) —
  The audio token id.
* **feature\_layer** (`int`, *optional*, defaults to -2) —
  The index of the layer of the encoder from which to extract audio features.

This is the configuration class to store the configuration of a [Phi4MultimodalAudioModel](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalAudioModel). It is used to instantiate a
Phi4Multimodal audio encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the audio encoder of
[microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Phi4MultimodalAudioConfig

>>> # Initializing a Phi4MultimodalAudioConfig with microsoft/Phi-4-multimodal-instruct style configuration
>>> configuration = Phi4MultimodalAudioConfig()
```

## Phi4MultimodalVisionConfig

### class transformers.Phi4MultimodalVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L26)

( hidden\_size = 1152 intermediate\_size = 4304 num\_hidden\_layers = 27 num\_attention\_heads = 16 num\_channels = 3 image\_size = 448 patch\_size = 14 hidden\_act = 'gelu\_pytorch\_tanh' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 crop\_size: int = 448 image\_token\_id: int = 200010 feature\_layer: int = -2 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1152) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 4304) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 27) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input images.
* **image\_size** (`int`, *optional*, defaults to 448) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **crop\_size** (`int`, *optional*, defaults to 448) —
  Crop size for the input images.
* **image\_token\_id** (`int`, *optional*, defaults to 200010) —
  The image token id.
* **feature\_layer** (`int`, *optional*, defaults to -2) —
  The index of the layer of the encoder from which to extract image features.

This is the configuration class to store the configuration of a [Phi4MultimodalVisionModel](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalVisionModel). It is used to instantiate a
Phi4Multimodal vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the vision encoder of
[microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Phi4MultimodalVisionConfig

>>> # Initializing a Phi4MultimodalVisionConfig with microsoft/Phi-4-multimodal-instruct style configuration
>>> configuration = Phi4MultimodalVisionConfig()
```

## Phi4MultimodalConfig

### class transformers.Phi4MultimodalConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L244)

( vocab\_size = 200064 hidden\_size = 3072 intermediate\_size = 8192 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = 8 resid\_pdrop = 0.0 embd\_pdrop = 0.0 attention\_dropout = 0.0 hidden\_act = 'silu' max\_position\_embeddings = 131072 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None partial\_rotary\_factor = 1 bos\_token\_id = 199999 eos\_token\_id = [199999, 200020] pad\_token\_id = 199999 original\_max\_position\_embeddings = 4096 sliding\_window = None vision\_config = None audio\_config = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 200064) —
  Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Phi3Model](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Model).
* **hidden\_size** (`int`, *optional*, defaults to 3072) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 8192) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
* **resid\_pdrop** (`float`, *optional*, defaults to 0.0) —
  Dropout probability for mlp outputs.
* **embd\_pdrop** (`int`, *optional*, defaults to 0.0) —
  The dropout ratio for the embeddings.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio after computing the attention scores.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 131072) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon value used for the RMSNorm.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`dict`, *optional*) —
  The scaling strategy for the RoPE embeddings. If `None`, no scaling is applied. If a dictionary, it must
  contain the following keys: `type`, `short_factor` and `long_factor`. The `type` must be `longrope` and
  the `short_factor` and `long_factor` must be lists of numbers with the same length as the hidden size
  divided by the number of attention heads divided by 2.
* **partial\_rotary\_factor** (`float`, *optional*, defaults to `1.0`) —
  Percentage of the query and keys which will have rotary embedding. Must be between 0.0 and 1.0.
* **bos\_token\_id** (`int`, *optional*, defaults to 199999) —
  The id of the “beginning-of-sequence” token.
* **eos\_token\_id** (`int` or `list[int]`, *optional*, defaults to `[199999, 200020]`) —
  The id of the “end-of-sequence” token.
* **pad\_token\_id** (`int`, *optional*, defaults to 199999) —
  The id of the padding token.
* **original\_max\_position\_embeddings** (`int`, *optional*, defaults to 4096) —
  The maximum sequence length that this model was trained with. This is used to determine the size of the
  original RoPE embeddings when using long scaling.
* **sliding\_window** (`int`, *optional*) —
  Sliding window attention window size. If `None`, no sliding window is applied.
* **vision\_config** (`Phi4MultimodalVisionConfig` or `dict`, *optional*) —
  The vision config for the underlying image embedding model. If not provided, will default to the configuration
  used to instantiate a model similar in architecture as
  [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).
* **audio\_config** (`Phi4MultimodalAudioConfig` or `dict`, *optional*) —
  The audio config for the underlying audio embedding model. If not provided, will default to the configuration
  used to instantiate a model similar in architecture as
  [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).

This is the configuration class to store the configuration of a [Phi4MultimodalModel](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalModel). It is used to instantiate a
Phi4Multimodal model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
[microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Phi4MultimodalModel, Phi4MultimodalConfig

>>> # Initializing a Phi4Multimodal style configuration
>>> configuration = Phi4MultimodalConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

>>> # Initializing a model from the configuration
>>> model = Phi4MultimodalModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Phi4MultimodalAudioModel

### class transformers.Phi4MultimodalAudioModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1061)

( config: Phi4MultimodalAudioConfig  )

#### forward\_embeddings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1097)

( hidden\_states masks  )

Forwarding the inputs through the top embedding layers

## Phi4MultimodalVisionModel

### class transformers.Phi4MultimodalVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L527)

( config: Phi4MultimodalVisionConfig  )

## Phi4MultimodalModel

### class transformers.Phi4MultimodalModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1606)

( config: Phi4MultimodalConfig  )

Parameters

* **config** ([Phi4MultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Phi4 Multimodal Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1628)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None image\_pixel\_values: typing.Optional[torch.FloatTensor] = None image\_sizes: typing.Optional[torch.LongTensor] = None image\_attention\_mask = None audio\_input\_features: typing.Optional[torch.FloatTensor] = None audio\_embed\_sizes = None audio\_attention\_mask = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs  )

image\_pixel\_values (`torch.FloatTensor`, *optional*):
If the input contains images, these correspond to the pixel values after transformations (as returned by
the Processor)
image\_sizes (`torch.LongTensor`, *optional*):
If the input contains images, these correspond to size of each image.
image\_attention\_mask (`torch.LongTensor`, *optional*):
Attention mask for the images.
audio\_input\_features (`torch.FloatTensor`, *optional*):
If the input contains audio samples, these correspond to the values after transformation (as returned by
the Processor).
audio\_embed\_sizes (`torch.Tensor`, *optional*):
Size of the audio inputs.
audio\_attention\_mask (`torch.Tensor, *optional*):
Attention mask for the audio inputs.

## Phi4MultimodalForCausalLM

### class transformers.Phi4MultimodalForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1724)

( config  )

Parameters

* **config** ([Phi4MultimodalForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Phi4 Multimodal Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1738)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None image\_pixel\_values: typing.Optional[torch.FloatTensor] = None image\_sizes: typing.Optional[torch.LongTensor] = None image\_attention\_mask = None audio\_input\_features: typing.Optional[torch.FloatTensor] = None audio\_embed\_sizes = None audio\_attention\_mask = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **image\_pixel\_values** (`torch.FloatTensor`, *optional*) —
  If the input contains images, these correspond to the pixel values after transformations (as returned by
  the Processor)
* **image\_sizes** (`torch.LongTensor`, *optional*) —
  If the input contains images, these correspond to size of each image.
* **image\_attention\_mask** (`torch.LongTensor`, *optional*) —
  Attention mask for the images.
* **audio\_input\_features** (`torch.FloatTensor`, *optional*) —
  If the input contains audio samples, these correspond to the values after transformation (as returned by
  the Processor).
* **audio\_embed\_sizes** (`torch.Tensor`, *optional*) —
  Size of the audio inputs.
* **audio\_attention\_mask** (`torch.Tensor, *optional*) —
  Attention mask for the audio inputs.
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
elements depending on the configuration ([Phi4MultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig)) and inputs.

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

The [Phi4MultimodalForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, Phi4MultimodalForCausalLM
>>> model = Phi4MultimodalForCausalLM.from_pretrained("TBA")
>>> tokenizer = AutoTokenizer.from_pretrained("TBA")
>>> prompt = "This is an example script ."
>>> inputs = tokenizer(prompt, return_tensors="pt")
>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/phi4_multimodal.md)
