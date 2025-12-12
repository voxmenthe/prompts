*This model was released on 2025-04-14 and added to Hugging Face Transformers on 2025-04-18.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

# InternVL

The InternVL3 family of Visual Language Models was introduced in [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://huggingface.co/papers/2504.10479).

The abstract from the paper is the following:

*We introduce InternVL3, a significant advancement in the InternVL series featuring a native multimodal pre-training paradigm. Rather than adapting a text-only large language model (LLM) into a multimodal large language model (MLLM) that supports visual inputs, InternVL3 jointly acquires multimodal and linguistic capabilities from both diverse multimodal data and pure-text corpora during a single pre-training stage. This unified training paradigm effectively addresses the complexities and alignment challenges commonly encountered in conventional post-hoc training pipelines for MLLMs. To further improve performance and scalability, InternVL3 incorporates variable visual position encoding (V2PE) to support extended multimodal contexts, employs advanced post-training techniques such as supervised fine-tuning (SFT) and mixed preference optimization (MPO), and adopts test-time scaling strategies alongside an optimized training infrastructure. Extensive empirical evaluations demonstrate that InternVL3 delivers superior performance across a wide range of multi-modal tasks. In particular, InternVL3-78B achieves a score of 72.2 on the MMMU benchmark, setting a new state-of-the-art among open-source MLLMs. Its capabilities remain highly competitive with leading proprietary models, including ChatGPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro, while also maintaining strong pure-language proficiency. In pursuit of open-science principles, we will publicly release both the training data and model weights to foster further research and development in next-generation MLLMs.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/internvl_architecture.png) Overview of InternVL3 models architecture, which is the same as InternVL2.5. Taken from the [original checkpoint.](https://huggingface.co/OpenGVLab/InternVL3-1B) ![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/internvl_overview_performance.png) Comparison of InternVL3 performance on OpenCompass against other SOTA VLLMs. Taken from the [original checkpoint.](https://huggingface.co/OpenGVLab/InternVL3-1B)

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/OpenGVLab/InternVL).

## Usage example

### Inference with Pipeline

Here is how you can use the `image-text-to-text` pipeline to perform inference with the `InternVL3` models in just a few lines of code:


```
>>> from transformers import pipeline

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
...             },
...             {"type": "text", "text": "Describe this image."},
...         ],
...     },
... ]

>>> pipe = pipeline("image-text-to-text", model="OpenGVLab/InternVL3-1B-hf")
>>> outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
>>> outputs[0]["generated_text"]
'The image showcases a vibrant scene of nature, featuring several flowers and a bee. \n\n1. **Foreground Flowers**: \n   - The primary focus is on a large, pink cosmos flower with a prominent yellow center. The petals are soft and slightly r'
```

### Inference on a single image

This example demonstrates how to perform inference on a single image with the InternVL models using chat templates.

> [!NOTE]
> Note that the model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.


```
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
...             {"type": "text", "text": "Please describe the image explicitly."},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> generate_ids = model.generate(**inputs, max_new_tokens=50)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> decoded_output
'The image shows two cats lying on a pink blanket. The cat on the left is a tabby with a mix of brown, black, and white fur, and it appears to be sleeping with its head resting on the blanket. The cat on the'
```

### Text-only generation

This example shows how to generate text using the InternVL model without providing any image input.


```
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "text", "text": "Write a haiku"},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device, dtype=torch.bfloat16)

>>> generate_ids = model.generate(**inputs, max_new_tokens=50)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> print(decoded_output)
"Whispers of dawn,\nSilent whispers of the night,\nNew day's light begins."
```

### Batched image and text inputs

InternVL models also support batched image and text inputs.


```
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
...                 {"type": "text", "text": "Describe this image"},
...             ],
...         },
...     ],
... ]


>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["user\n\nWrite a haiku for this image\nassistant\nSilky lake,  \nWooden pier,  \nNature's peace.",
 'user\n\nDescribe this image\nassistant\nThe image shows a street scene with a traditional Chinese archway, known as a "Chinese Gate" or "Chinese Gate of']
```

### Batched multi-image input

This implementation of the InternVL models supports batched text-images inputs with different number of images for each text.


```
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
...                 {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
...                 {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...             ],
...         },
...     ],
>>> ]

>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["user\n\nWrite a haiku for this image\nassistant\nSilky lake,  \nWooden pier,  \nNature's peace.",
 'user\n\n\nThese images depict two different landmarks. Can you identify them?\nassistant\nYes, these images depict the Statue of Liberty and the Golden Gate Bridge.']
```

### Video input

InternVL models can also handle video inputs. Here is an example of how to perform inference on a video input using chat templates.


```
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

>>> model_checkpoint = "OpenGVLab/InternVL3-8B-hf"
>>> quantization_config = BitsAndBytesConfig(load_in_4bit=True)
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, quantization_config=quantization_config)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "video",
...                 "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
...             },
...             {"type": "text", "text": "What type of shot is the man performing?"},
...         ],
...     }
>>> ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     return_tensors="pt",
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     num_frames=8,
>>> ).to(model.device, dtype=torch.float16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
>>> decoded_output
'The man is performing a forehand shot.'
```

### Interleaved image and video inputs

This example showcases how to handle a batch of chat conversations with interleaved image and video inputs using chat template.


```
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
>>> import torch

>>> model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
...                 {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
...                 {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
...                 {"type": "text", "text": "What type of shot is the man performing?"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
>>> ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     padding=True,
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
>>> ).to(model.device, dtype=torch.bfloat16)

>>> outputs = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
>>> decoded_outputs
['user\n\n\nThese images depict two different landmarks. Can you identify them?\nassistant\nThe images depict the Statue of Liberty and the Golden Gate Bridge.',
 'user\nFrame1: \nFrame2: \nFrame3: \nFrame4: \nFrame5: \nFrame6: \nFrame7: \nFrame8: \nWhat type of shot is the man performing?\nassistant\nA forehand shot',
 "user\n\nWrite a haiku for this image\nassistant\nSilky lake,  \nWooden pier,  \nNature's peace."]
```

## InternVLVisionConfig

### class transformers.InternVLVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/configuration_internvl.py#L21)

( hidden\_size = 1024 num\_hidden\_layers = 24 num\_attention\_heads = 16 attention\_bias = False use\_qk\_norm = False intermediate\_size = 4096 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_dropout = 0.0 projection\_dropout = 0.0 initializer\_range = 0.02 norm\_type = 'layer\_norm' layer\_norm\_eps = 1e-06 image\_size = [448, 448] patch\_size = [14, 14] num\_channels = 3 use\_mask\_token = False use\_absolute\_position\_embeddings = True layer\_scale\_init\_value = 0.1 use\_mean\_pooling = True \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the queries, keys and values.
* **use\_qk\_norm** (`bool`, *optional*, defaults to `False`) —
  Whether to apply normalization to the queries and keys before the attention operation.
* **intermediate\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  Dropout probability for attention weights.
* **projection\_dropout** (`float`, *optional*, defaults to 0.0) —
  Dropout probability for the projection layer.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **norm\_type** (`str`, *optional*, defaults to `"layer_norm"`) —
  The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`int` or `list[int]`, *optional*, defaults to `[448, 448]`) —
  The size (resolution) of each image.
* **patch\_size** (`int` or `list[int]`, *optional*, defaults to `[14, 14]`) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **use\_mask\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to use a mask token for masked image modeling.
* **use\_absolute\_position\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether to use BERT-style absolute position embeddings.
* **layer\_scale\_init\_value** (`float`, *optional*, defaults to 0.1) —
  Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
* **use\_mean\_pooling** (`bool`, *optional*, defaults to `True`) —
  Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
  CLS token, before applying the classification head.

This is the configuration class to store the configuration of a [InternVLVisionModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionModel). It is used to instantiate an InternVLVisionModel
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
a similar configuration to that of the InternVL3-1B.
e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Example:


```
>>> from transformers import InternVLVisionConfig, InternVLVisionModel

>>> # Initializing a InternVLVisionModel OpenGVLab/InternVL3-1B-hf style configuration
>>> configuration = InternVLVisionConfig()

>>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
>>> model = InternVLVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## InternVLConfig

### class transformers.InternVLConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/configuration_internvl.py#L142)

( vision\_config = None text\_config = None image\_token\_id = 151667 image\_seq\_length = 256 downsample\_ratio = 0.5 projector\_hidden\_act = 'gelu' vision\_feature\_layer = -1 vision\_feature\_select\_strategy = 'default' \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `InternVisonConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`) —
  The config object or dictionary of the text backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 151667) —
  The image token index to encode the image prompt.
* **image\_seq\_length** (`int`, *optional*, defaults to 256) —
  Number of image tokens to use per image patch.
* **downsample\_ratio** (`float`, *optional*, defaults to 0.5) —
  Factor by which to downsample the image.
* **projector\_hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the projector.
* **vision\_feature\_layer** (`int`, *optional*, defaults to -1) —
  The index of the layer to use as the image features.
* **vision\_feature\_select\_strategy** (`str`, *optional*, defaults to `"default"`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.

This is the configuration class to store the configuration of a [InternVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLForConditionalGeneration). It is used to instantiate a
InternVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of InternVL3-1B.
e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import InternVLForConditionalGeneration, InternVLConfig

>>> # Initializing a InternVL style configuration
>>> configuration = InternVLConfig()

>>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
>>> model = InternVLForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## InternVLVisionModel

### class transformers.InternVLVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/modeling_internvl.py#L445)

( config: InternVLVisionConfig  )

Parameters

* **config** ([InternVLVisionConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Internvl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/modeling_internvl.py#L463)

( pixel\_values: Tensor bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([InternVLProcessor](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLProcessor) uses
  `image_processor_class` for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or `tuple(torch.FloatTensor)`

A `transformers.models.internvl.modeling_internvl.InternVLVisionModelOutputWithPooling` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a
  Linear layer and a Tanh activation function.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [InternVLVisionModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## InternVLModel

### class transformers.InternVLModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/modeling_internvl.py#L558)

( config: InternVLConfig  )

Parameters

* **config** ([InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The InternVL model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/modeling_internvl.py#L659)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([InternVLProcessor](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLProcessor) uses
  `image_processor_class` for processing images).
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

Returns

`transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.internvl.modeling_internvl.InternVLModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

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

The [InternVLModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## InternVLForConditionalGeneration

### class transformers.InternVLForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/modeling_internvl.py#L803)

( config: InternVLConfig  )

Parameters

* **config** ([InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The INTERNVL model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/modeling_internvl.py#L860)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 image\_sizes: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([InternVLProcessor](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLProcessor) uses
  `image_processor_class` for processing images).
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

`transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.internvl.modeling_internvl.InternVLCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig)) and inputs.

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

The [InternVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> torch_device = "cuda"
>>> processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
>>> model = AutoModelForImageTextToText.from_pretrained(
...     "OpenGVLab/InternVL3-1B-hf", dtype=torch.bfloat16, device_map=torch_device
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
...             },
...             {
...                 "type": "image",
...                 "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
...             },
...             {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
>>> generate_ids = model.generate(**inputs, max_new_tokens=200)
>>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
The images depict the Statue of Liberty and the Golden Gate Bridge.
```

## InternVLProcessor

### class transformers.InternVLProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/processing_internvl.py#L47)

( image\_processor = None tokenizer = None video\_processor = None image\_seq\_length: int = 256 chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*) —
  The tokenizer is a required input.
* **video\_processor** ([AutoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoVideoProcessor), *optional*) —
  The video processor is a required input.
* **image\_seq\_length** (`int`, *optional*, defaults to 256) —
  The number of image token to use per image patch. it should be set so that:
  image\_seq\_length = (config.image\_size // config.patch\_size)  **2 \* (config.scale\_factor**2)
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a InternVL processor which wraps a [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor) and
`PretrainedTokenizerFast` tokenizer into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## InternVLVideoProcessor

### class transformers.InternVLVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/video_processing_internvl.py#L57)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.internvl.video\_processing\_internvl.InternVLVideoProcessorInitKwargs]  )

#### sample\_frames

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/internvl/video_processing_internvl.py#L74)

( metadata: VideoMetadata num\_frames: typing.Optional[int] = None fps: typing.Union[int, float, NoneType] = None initial\_shift: typing.Union[bool, float, int, NoneType] = None \*\*kwargs  ) → np.ndarray

Parameters

* **metadata** (`VideoMetadata`) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **num\_frames** (`int`, *optional*) —
  Maximum number of frames to sample. Defaults to `self.num_frames`.
* **fps** (`int` or `float`, *optional*) —
  Target frames to sample per second. Defaults to `self.fps`.
* **initial\_shift** (`bool`, `float` or `int`, defaults to `self.initial_shift`) —
  The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.

Returns

np.ndarray

Indices to sample video frames.

Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
and `fps` are mutually exclusive.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/internvl.md)
