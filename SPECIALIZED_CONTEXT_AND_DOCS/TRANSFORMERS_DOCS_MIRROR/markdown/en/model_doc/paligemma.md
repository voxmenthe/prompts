*This model was released on 2024-07-10 and added to Hugging Face Transformers on 2024-05-14.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# PaliGemma

[PaliGemma](https://huggingface.co/papers/2407.07726) is a family of vision-language models (VLMs), combining [SigLIP](./siglip) with the [Gemma](./gemma) 2B model. PaliGemma is available in 3B, 10B, and 28B parameters. The main purpose of PaliGemma is to provide an adaptable base VLM that is easy to transfer to other tasks. The SigLIP vision encoder is a “shape optimized” contrastively pretrained [ViT](./vit) that converts an image into a sequence of tokens and prepended to an optional prompt. The Gemma 2B model is used as the decoder. PaliGemma uses full attention on all image and text tokens to maximize its capacity.

[PaliGemma 2](https://huggingface.co/papers/2412.03555) improves on the first model by using Gemma 2 (2B, 9B, and 27B parameter variants) as the decoder. These are available as **pt** or **mix** variants. The **pt** checkpoints are intended for further fine-tuning and the **mix** checkpoints are ready for use out of the box.

You can find all the original PaliGemma checkpoints under the [PaliGemma](https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda), [PaliGemma 2](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48), and [PaliGemma 2 Mix](https://huggingface.co/collections/google/paligemma-2-mix-67ac6a251aaf3ee73679dcc4) collections.

Click on the PaliGemma models in the right sidebar for more examples of how to apply PaliGemma to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/paligemma2-3b-mix-224",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="What is in this image?"
)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
# pip install torchao
import torch
import requests
from PIL import Image
from transformers import TorchAoConfig, AutoProcessor, PaliGemmaForConditionalGeneration

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-28b-mix-224",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    "google/paligemma2-28b-mix-224",
)

prompt = "What is in this image?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.


```
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("google/paligemma2-3b-mix-224")
visualizer("<img> What is in this image?")
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/paligemma2-attn-mask.png)

## Notes

* PaliGemma is not a conversational model and works best when fine-tuned for specific downstream tasks such as image captioning, visual question answering (VQA), object detection, and document understanding.
* [PaliGemmaProcessor](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor) can prepare images, text, and optional labels for the model. Pass the `suffix` parameter to the processor to create labels for the model during fine-tuning.


  ```
  prompt = "What is in this image?"
  answer = "a pallas cat"
  inputs = processor(images=image, text=prompt, suffix=answer, return_tensors="pt")
  ```
* PaliGemma can support multiple input images if it is fine-tuned to accept multiple images. For example, the [NLVR2](https://huggingface.co/google/paligemma-3b-ft-nlvr2-448) checkpoint supports multiple images. Pass the images as a list to the processor.


  ```
  import torch
  import requests
  from PIL import Image
  from transformers import TorchAoConfig, AutoProcessor, PaliGemmaForConditionalGeneration

  model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-ft-nlvr2-448")
  processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-nlvr2-448")

  prompt = "Are these two images the same?"
  cat_image = Image.open(
      requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", stream=True).raw
  )
  cow_image = Image.open(
      requests.get(
          "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4=", stream=True
      ).raw
  )

  inputs = processor(images=[[cat_image, cow_image]], text=prompt, return_tensors="pt")

  output = model.generate(**inputs, max_new_tokens=20, cache_implementation="static")
  print(processor.decode(output[0], skip_special_tokens=True))
  ```

## PaliGemmaConfig

### class transformers.PaliGemmaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/configuration_paligemma.py#L24)

( vision\_config = None text\_config = None image\_token\_index = 256000 vocab\_size = 257152 projection\_dim = 2048 hidden\_size = 2048 \*\*kwargs  )

Parameters

* **vision\_config** (`PaliGemmaVisionConfig`, *optional*) —
  Custom vision config or dict
* **text\_config** (`Union[AutoConfig, dict]`, *optional*) —
  The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
* **image\_token\_index** (`int`, *optional*, defaults to 256000) —
  The image token index to encode the image prompt.
* **vocab\_size** (`int`, *optional*, defaults to 257152) —
  Vocabulary size of the PaliGemmamodel. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [~PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration)
* **projection\_dim** (`int`, *optional*, defaults to 2048) —
  Dimension of the multimodal projection space.
* **hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the hidden layer of the Language model.

This is the configuration class to store the configuration of a [PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration). It is used to instantiate an
PaliGemmamodel according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the PaliGemma-2B.

e.g. [paligemma-hf/paligemma-2b](https://huggingface.co/paligemma-hf/paligemma-2b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

>>> # Initializing a Siglip-like vision config
>>> vision_config = SiglipVisionConfig()

>>> # Initializing a PaliGemma config
>>> text_config = GemmaConfig()

>>> # Initializing a PaliGemma paligemma-3b-224 style configuration
>>> configuration = PaliGemmaConfig(vision_config, text_config)

>>> # Initializing a model from the paligemma-3b-224 style configuration
>>> model = PaliGemmaForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PaliGemmaProcessor

### class transformers.PaliGemmaProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/processing_paligemma.py#L101)

( image\_processor = None tokenizer = None chat\_template = None \*\*kwargs  )

Parameters

* **image\_processor** ([SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

[PaliGemmaProcessor](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor) offers all the functionalities of [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) and [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## PaliGemmaModel

### class transformers.PaliGemmaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/modeling_paligemma.py#L138)

( config: PaliGemmaConfig  )

Parameters

* **config** ([PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Base Paligemma model which consists of a vision backbone and a language model withou language modeling head.,

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/modeling_paligemma.py#L275)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[list[torch.FloatTensor], transformers.cache\_utils.Cache, NoneType] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.paligemma.modeling_paligemma.PaligemmaModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor). See [SiglipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([PaliGemmaProcessor](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor) uses
  [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.
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

Returns

`transformers.models.paligemma.modeling_paligemma.PaligemmaModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.paligemma.modeling_paligemma.PaligemmaModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig)) and inputs.

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

The [PaliGemmaModel](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

>>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
>>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

>>> prompt = "Where is the cat standing?"
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs,)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Where is the cat standing?\nsnow"
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/modeling_paligemma.py#L235)

( pixel\_values: FloatTensor  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) —
  The tensors corresponding to the input images.

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_images, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/modeling_paligemma.py#L251)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## PaliGemmaForConditionalGeneration

### class transformers.PaliGemmaForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/modeling_paligemma.py#L392)

( config: PaliGemmaConfig  )

Parameters

* **config** ([PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Base Paligemma model which consists of a vision backbone and a language model without language modeling head.,

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/paligemma/modeling_paligemma.py#L435)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[list[torch.FloatTensor], transformers.cache\_utils.Cache, NoneType] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.paligemma.modeling_paligemma.PaliGemmaCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor). See [SiglipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([PaliGemmaProcessor](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor) uses
  [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.
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
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.paligemma.modeling_paligemma.PaliGemmaCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.paligemma.modeling_paligemma.PaliGemmaCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
  image\_hidden\_states of the model produced by the vision encoder after projecting last hidden state.

The [PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

>>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
>>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

>>> prompt = "Where is the cat standing?"
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs,)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Where is the cat standing?\nsnow"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/paligemma.md)
