![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Florence-2

[Florence-2](https://huggingface.co/papers/2311.06242) is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. It leverages the FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. The model’s sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

You can find all the original Florence-2 checkpoints under the [Florence-2](https://huggingface.co/models?other=florence-2) collection.

This model was contributed by [ducviet00](https://huggingface.co/ducviet00).
Click on the Florence-2 models in the right sidebar for more examples of how to apply Florence-2 to different vision and language tasks.

The example below demonstrates how to perform object detection with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
import requests
from PIL import Image
from transformers import pipeline

pipeline = pipeline(
    "image-text-to-text",
    model="ducviet00/Florence-2-base-hf",
    device=0,
    dtype=torch.bfloat16
)

pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    text="<OD>"
)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the model to 4-bit.


```
# pip install bitsandbytes
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = Florence2ForConditionalGeneration.from_pretrained(
    "microsoft/Florence-2-large",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

image_size = image.size
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)

print(parsed_answer)
```

![]()

## Notes

* Florence-2 is a prompt-based model. You need to provide a task prompt to tell the model what to do. Supported tasks are:
  + `<OCR>`
  + `<OCR_WITH_REGION>`
  + `<CAPTION>`
  + `<DETAILED_CAPTION>`
  + `<MORE_DETAILED_CAPTION>`
  + `<OD>`
  + `<DENSE_REGION_CAPTION>`
  + `<CAPTION_TO_PHRASE_GROUNDING>`
  + `<REFERRING_EXPRESSION_SEGMENTATION>`
  + `<REGION_TO_SEGMENTATION>`
  + `<OPEN_VOCABULARY_DETECTION>`
  + `<REGION_TO_CATEGORY>`
  + `<REGION_TO_DESCRIPTION>`
  + `<REGION_TO_OCR>`
  + `<REGION_PROPOSAL>`
* The raw output of the model is a string that needs to be parsed. The [Florence2Processor](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor) has a [post\_process\_generation()](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor.post_process_generation) method that can parse the string into a more usable format, like bounding boxes and labels for object detection.

## Resources

* [Florence-2 technical report](https://huggingface.co/papers/2311.06242)
* [Jupyter Notebook for inference and visualization of Florence-2-large model](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)

## Florence2VisionConfig

### class transformers.Florence2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/configuration_florence2.py#L29)

( in\_channels = 3 depths = (1, 1, 9, 1) patch\_size = (7, 3, 3, 3) patch\_stride = (4, 2, 2, 2) patch\_padding = (3, 1, 1, 1) patch\_prenorm = (False, True, True, True) embed\_dim = (128, 256, 512, 1024) num\_heads = (4, 8, 16, 32) num\_groups = (4, 8, 16, 32) window\_size = 12 drop\_path\_rate = 0.1 mlp\_ratio = 4.0 qkv\_bias = True activation\_function = 'gelu' projection\_dim = 1024 max\_temporal\_embeddings = 100 max\_position\_embeddings = 50 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **in\_channels** (`int`, *optional*, defaults to 3) —
  Number of input image channels.
* **depths** (`Tuple[int]`, *optional*, defaults to `(1, 1, 9, 1)`) —
  The depth of the model.
* **patch\_size** (`Tuple[int]`, *optional*, defaults to `(7, 3, 3, 3)`) —
  The patch size of the image.
* **patch\_stride** (`Tuple[int]`, *optional*, defaults to `(4, 2, 2, 2)`) —
  The patch stride of the image.
* **patch\_padding** (`Tuple[int]`, *optional*, defaults to `(3, 1, 1, 1)`) —
  The patch padding of the image.
* **patch\_prenorm** (`Tuple[bool]`, *optional*, defaults to `(False, True, True, True)`) —
  Whether to apply layer normalization before the patch embedding layer.
* **embed\_dim** (`Tuple[int]`, *optional*, defaults to `(128, 256, 512, 1024)`) —
  The dimension of the embedding layer.
* **num\_heads** (`Tuple[int]`, *optional*, defaults to `(4, 8, 16, 32)`) —
  The number of attention heads.
* **num\_groups** (`Tuple[int]`, *optional*, defaults to `(4, 8, 16, 32)`) —
  The number of groups.
* **window\_size** (`int`, *optional*, defaults to 12) —
  The window size of the model.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.1) —
  The dropout rate of the drop path layer.
* **mlp\_ratio** (`int`, *optional*, defaults to 4.0) —
  Ratio of mlp hidden dim to embedding dim.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  If True, add a learnable bias to query, key, value.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **projection\_dim** (`int`, *optional*, defaults to 1024) —
  The dimension of the projection layer.
* **max\_temporal\_embeddings** (`int`, *optional*, defaults to 100) —
  The configuration of the visual temporal embedding.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 50) —
  The configuration of the image position embedding.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a `Florence2VisionModel`. It is used to instantiate a Florence2VisionModel
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Florence2VisionModel architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Florence2VisionConfig, Florence2VisionModel

>>> # Initializing a Florence2 Vision style configuration
>>> configuration = Florence2VisionConfig()

>>> # Initializing a model (with random weights)
>>> model = Florence2VisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Florence2Config

### class transformers.Florence2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/configuration_florence2.py#L137)

( text\_config = None vision\_config = None image\_token\_id = 51289 is\_encoder\_decoder = True \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [AutoConfig](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Florence2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2VisionConfig).
* **image\_token\_id** (`int`, *optional*, defaults to 51289) —
  The image token index to encode the image prompt.
* **is\_encoder\_decoder** (bool, optional, *optional*, defaults to `True`) —
  Whether the model is used as an encoder/decoder or not.

This is the configuration class to store the configuration of a [Florence2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2ForConditionalGeneration). It is used to instantiate an
Florence-2 model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the Florence-2
[microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Florence2ForConditionalGeneration, Florence2Config, CLIPVisionConfig, BartConfig

>>> # Initializing a clip-like vision config
>>> vision_config = CLIPVisionConfig()

>>> # Initializing a Bart config
>>> text_config = BartConfig()

>>> # Initializing a Florence-2 configuration
>>> configuration = Florence2Config(vision_config, text_config)

>>> # Initializing a model from the florence-2 configuration
>>> model = Florence2ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Florence2Processor

### class transformers.Florence2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/processing_florence2.py#L47)

( image\_processor = None tokenizer = None num\_additional\_image\_tokens: int = 0 post\_processor\_config: typing.Optional[dict] = None \*\*kwargs  )

Parameters

* **image\_processor** (`AutoImageProcessor`, *optional*) —
  The image processor is a required input.
* **tokenizer** (`Union[BartTokenizer, BartTokenizerFast]`, *optional*) —
  The tokenizer is a required input.
* **num\_additional\_image\_tokens** (`int`, *optional*, defaults to 0) —
  Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
  extra tokens appended, no need to set this arg.
* **post\_processor\_config** (`dict`, *optional*, defaults to 0) —
  Task-specific parsing rules for `Florence2PostProcessor`, e.g. regex patterns,
  thresholds, or banned tokens.

Constructs a Florence2 processor which wraps a Florence2 image processor and a Florence2 tokenizer into a single processor.

[Florence2Processor](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor) offers all the functionalities of [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor) and [BartTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor.decode) for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/processing_florence2.py#L247)

( \*args \*\*kwargs  )

This method forwards all its arguments to BartTokenizerFast’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/processing_florence2.py#L254)

( \*args \*\*kwargs  )

This method forwards all its arguments to BartTokenizerFast’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

#### post\_process\_generation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/processing_florence2.py#L307)

( text = None sequence = None task = None image\_size = None  ) → `Dict[str, Any]`

Parameters

* **text** (`str`, *optional*) —
  Generated text.
* **sequence** (`Union[List[int], torch.Tensor]`, *optional*) —
  Generated token sequence.
* **task** (`str`, *optional*) —
  The task for post-processing.
* **image\_size** (`Tuple[int, int]`, *optional*) —
  Image size for dequantization.

Returns

`Dict[str, Any]`

Post-processed results keyed by task.

Post-process generation outputs based on the task.

#### post\_process\_image\_text\_to\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/processing_florence2.py#L289)

( generated\_outputs skip\_special\_tokens = False \*\*kwargs  ) → `list[str]`

Parameters

* **generated\_outputs** (`torch.Tensor` or `np.ndarray`) —
  The output of the model. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
  containing the token ids of the generated sequences.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the output. Argument passed to the tokenizer’s `batch_decode` method.
* \***\*kwargs** —
  Additional arguments to be passed to the tokenizer’s `batch_decode method`.

Returns

`list[str]`

The decoded text output.

Post-processes the output of `FuyuForConditionalGeneration` to only return the text output.

## Florence2Model

### class transformers.Florence2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/modeling_florence2.py#L643)

( config: Florence2Config  )

Parameters

* **config** ([Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Florence-2 is a vision model for captioning, detection, and segmentation.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/modeling_florence2.py#L708)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[list[torch.FloatTensor]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.florence2.modeling_florence2.Florence2Seq2SeqModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Florence2Processor](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor) uses
  `image_processor_class` for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **encoder\_outputs** (`list[torch.FloatTensor]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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

`transformers.models.florence2.modeling_florence2.Florence2Seq2SeqModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.florence2.modeling_florence2.Florence2Seq2SeqModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`~cache_utils.EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [Florence2Model](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Florence2ForConditionalGeneration

### class transformers.Florence2ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/modeling_florence2.py#L804)

( config: Florence2Config  )

Parameters

* **config** ([Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Florence-2 is a vision model for captioning, detection, and segmentation.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/modeling_florence2.py#L849)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[list[torch.FloatTensor]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.florence2.modeling_florence2.Florence2Seq2SeqLMOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Florence2Processor](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor) uses
  `image_processor_class` for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`list[torch.FloatTensor]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
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

`transformers.models.florence2.modeling_florence2.Florence2Seq2SeqLMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.florence2.modeling_florence2.Florence2Seq2SeqLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`~cache_utils.EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [Florence2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Florence2ForConditionalGeneration

>>> model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large")
>>> processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

>>> prompt = "<CAPTION>"
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_length=100)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"A green car parked in front of a yellow building."
```

## Florence2VisionBackbone

### class transformers.Florence2VisionBackbone

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/modeling_florence2.py#L503)

( config: Florence2VisionConfig  )

Parameters

* **config** ([Florence2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2VisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Florence2 backbone.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/florence2/modeling_florence2.py#L550)

( hidden\_states: Tensor  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/florence2.md)
