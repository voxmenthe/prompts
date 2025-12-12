*This model was released on 2025-05-13 and added to Hugging Face Transformers on 2025-03-04.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Aya Vision

[Aya Vision](https://huggingface.co/papers/2505.08751) is a family of open-weight multimodal vision-language models from Cohere Labs. It is trained with a synthetic annotation framework that generates high-quality multilingual image captions, improving Aya Vision’s generated responses. In addition, a cross-modal model merging technique is used to prevent the model from losing its text capabilities after adding vision capabilities. The model combines a CommandR-7B language model with a SigLIP vision encoder.

You can find all the original Aya Vision checkpoints under the [Aya Vision](https://huggingface.co/collections/CohereLabs/cohere-labs-aya-vision-67c4ccd395ca064308ee1484) collection.

This model was contributed by [saurabhdash](https://huggingface.co/saurabhdash) and [yonigozlan](https://huggingface.co/yonigozlan).

Click on the Aya Vision models in the right sidebar for more examples of how to apply Aya Vision to different image-to-text tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
from transformers import pipeline

pipe = pipeline(model="CohereLabs/aya-vision-8b", task="image-text-to-text", device_map="auto")

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo="},
        {"type": "text", "text": "Bu resimde hangi anıt gösterilmektedir?"},
    ]},
    ]
outputs = pipe(text=messages, max_new_tokens=300, return_full_text=False)

print(outputs)
```

Quantization reduces the memory footprint of large models by representing weights at lower precision. Refer to the [Quantization](../quantization/overview) overview for supported backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.


```
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

processor = AutoProcessor.from_pretrained("CohereLabs/aya-vision-32b", use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    "CohereLabs/aya-vision-32b",
    quantization_config=bnb_config,
    device_map="auto"
)

inputs = processor.apply_chat_template(
    [
    {"role": "user", "content": [
        {"type": "image", "url": "https://huggingface.co/roschmid/dog-races/resolve/main/images/Border_Collie.jpg"},
        {"type": "text",  "text":"Describe what you see."}
    ]}
    ],
    padding=True,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=50)
print(processor.tokenizer.decode(generated[0], skip_special_tokens=True))
```

## Notes

* Images are represented with the `<image>` tag in the chat template.
* Use the [apply\_chat\_template()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template) method to correctly format inputs.
* The example below demonstrates inference with multiple images.


  ```
  import torch
  from transformers import AutoProcessor, AutoModelForImageTextToText
      
  processor = AutoProcessor.from_pretrained("CohereForAI/aya-vision-8b")
  model = AutoModelForImageTextToText.from_pretrained(
      "CohereForAI/aya-vision-8b", device_map="auto", dtype=torch.float16
  )

  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
              },
              {
                  "type": "image",
                  "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
              },
              {
                  "type": "text",
                  "text": "These images depict two different landmarks. Can you identify them?",
              },
          ],
      },
  ]

  inputs = processor.apply_chat_template(
      messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
  ).to(model.device)

  gen_tokens = model.generate(
      **inputs, 
      max_new_tokens=300, 
      do_sample=True, 
      temperature=0.3,
  )

  gen_text = processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
  print(gen_text)
  ```
* The example below demonstrates inference with batched inputs.


  ```
  import torch
  from transformers import AutoProcessor, AutoModelForImageTextToText
      
  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForImageTextToText.from_pretrained(
      "CohereForAI/aya-vision-8b", device_map="auto", dtype=torch.float16
  )

  batch_messages = [
      [
          {
              "role": "user",
              "content": [
                  {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                  {"type": "text", "text": "Write a haiku for this image"},
              ],
          },
      ],
      [
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                  },
                  {
                      "type": "image",
                      "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                  },
                  {
                      "type": "text",
                      "text": "These images depict two different landmarks. Can you identify them?",
                  },
              ],
          },
      ],
  ]

  batch_inputs = processor.apply_chat_template(
      batch_messages, 
      padding=True, 
      add_generation_prompt=True, 
      tokenize=True, 
      return_dict=True, 
      return_tensors="pt"
  ).to(model.device)

  batch_outputs = model.generate(
      **batch_inputs,
      max_new_tokens=300,
      do_sample=True,
      temperature=0.3,
  )

  for i, output in enumerate(batch_outputs):
      response = processor.tokenizer.decode(
          output[batch_inputs.input_ids.shape[1]:], 
          skip_special_tokens=True
      )
      print(f"Response {i+1}:\n{response}\n")
  ```

## AyaVisionProcessor

### class transformers.AyaVisionProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/processing_aya_vision.py#L46)

( image\_processor = None tokenizer = None patch\_size: int = 28 img\_size: int = 364 image\_token = '<image>' downsample\_factor: int = 1 start\_of\_img\_token = '<|START\_OF\_IMG|>' end\_of\_img\_token = '<|END\_OF\_IMG|>' img\_patch\_token = '<|IMG\_PATCH|>' img\_line\_break\_token = '<|IMG\_LINE\_BREAK|>' tile\_token = 'TILE' tile\_global\_token = 'TILE\_GLOBAL' chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*) —
  The tokenizer is a required input.
* **patch\_size** (`int`, *optional*, defaults to 28) —
  The size of image patches for tokenization.
* **img\_size** (`int`, *optional*, defaults to 364) —
  The size of the image to be tokenized. This should correspond to the size given to the image processor.
* **image\_token** (`str`, *optional*, defaults to `"<image>"`) —
  The token to be used to represent an image in the text.
* **downsample\_factor** (`int`, *optional*, defaults to 1) —
  The factor by which to scale the patch size.
* **start\_of\_img\_token** (`str`, *optional*, defaults to `"<|START_OF_IMG|>"`) —
  The token to be used to represent the start of an image in the text.
* **end\_of\_img\_token** (`str`, *optional*, defaults to `"<|END_OF_IMG|>"`) —
  The token to be used to represent the end of an image in the text.
* **img\_patch\_token** (`str`, *optional*, defaults to `"<|IMG_PATCH|>"`) —
  The token to be used to represent an image patch in the text.
* **img\_line\_break\_token** (`str`, *optional*, defaults to `"<|IMG_LINE_BREAK|>"`) —
  The token to be used to represent a line break in the text.
* **tile\_token** (`str`, *optional*, defaults to `"TILE"`) —
  The token to be used to represent an image patch in the text.
* **tile\_global\_token** (`str`, *optional*, defaults to `"TILE_GLOBAL"`) —
  The token to be used to represent the cover image in the text.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a AyaVision processor which wraps a [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor) and
`PretrainedTokenizerFast` tokenizer into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## AyaVisionConfig

### class transformers.AyaVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/configuration_aya_vision.py#L25)

( vision\_config = None text\_config = None vision\_feature\_select\_strategy = 'full' vision\_feature\_layer = -1 downsample\_factor = 2 adapter\_layer\_norm\_eps = 1e-06 image\_token\_index = 255036 \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `SiglipVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `Cohere2Config`) —
  The config object or dictionary of the text backbone.
* **vision\_feature\_select\_strategy** (`str`, *optional*, defaults to `"full"`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
  If `"full"`, the full vision features are used.
* **vision\_feature\_layer** (`int`, *optional*, defaults to -1) —
  The index of the layer to select the vision feature.
* **downsample\_factor** (`int`, *optional*, defaults to 2) —
  The downsample factor to apply to the vision features.
* **adapter\_layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon value used for layer normalization in the adapter.
* **image\_token\_index** (`int`, *optional*, defaults to 255036) —
  The image token index to encode the image prompt.

This is the configuration class to store the configuration of a [AyaVisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionForConditionalGeneration). It is used to instantiate an
AyaVision model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of AyaVision.
e.g. [CohereForAI/aya-vision-8b](https://huggingface.co/CohereForAI/aya-vision-8b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## AyaVisionModel

### class transformers.AyaVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L167)

( config: AyaVisionConfig  )

Parameters

* **config** ([AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The AyaVision model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L269)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.aya_vision.modeling_aya_vision.AyaVisionModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([AyaVisionProcessor](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionProcessor) uses
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.aya_vision.modeling_aya_vision.AyaVisionModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.aya_vision.modeling_aya_vision.AyaVisionModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig)) and inputs.

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

The [AyaVisionModel](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L190)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L245)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## AyaVisionForConditionalGeneration

### class transformers.AyaVisionForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L336)

( config: AyaVisionConfig  )

Parameters

* **config** ([AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The AYA\_VISION model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L393)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 image\_sizes: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.aya_vision.modeling_aya_vision.AyaVisionCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([AyaVisionProcessor](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionProcessor) uses
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

`transformers.models.aya_vision.modeling_aya_vision.AyaVisionCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.aya_vision.modeling_aya_vision.AyaVisionCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig)) and inputs.

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

The [AyaVisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, AyaVisionForConditionalGeneration
>>> import torch

>>> torch_device = "cuda:0"
>>> processor = AutoProcessor.from_pretrained("CohereForAI/aya-vision-8b", use_fast=True)
>>> model = AyaVisionForConditionalGeneration.from_pretrained("CohereForAI/aya-vision-8b", device_map=torch_device)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "url": "https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&name=medium",
...             },
...             {"type": "text", "text": "चित्र में लिखा पाठ क्या कहता है?"},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(
...     messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", device=torch_device
... ).to(model.device)

>>> gen_tokens = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.3)
>>> processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/aya_vision.md)
