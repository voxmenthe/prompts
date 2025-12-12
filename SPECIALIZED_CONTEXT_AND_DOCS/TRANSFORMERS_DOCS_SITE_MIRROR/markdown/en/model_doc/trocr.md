# TrOCR

[TrOCR](https://huggingface.co/papers/2109.10282) is a text recognition model for both image understanding and text generation. It doesn't require separate models for image processing or character generation. TrOCR is a simple single end-to-end system that uses a transformer to handle visual understanding and text generation.

You can find all the original TrOCR checkpoints under the [Microsoft](https://huggingface.co/microsoft/models?search=trocr) organization.

 TrOCR architecture. Taken from the original paper. 

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the TrOCR models in the right sidebar for more examples of how to apply TrOCR to different image and text tasks.

The example below demonstrates how to perform optical character recognition (OCR) with the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 8-bits.

```python
# pip install bitsandbytes accelerate
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, BitsandBytesConfig
import requests
from PIL import Image

# Set up the quantization configuration
quantization_config = BitsandBytesConfig(load_in_8bit=True)

# Use a large checkpoint for a more noticeable impact
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten",
    quantization_config=quantization_config
)

# load image from the IAM dataset
url = "[https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg](https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg)"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## Notes

- TrOCR wraps [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor)/[DeiTImageProcessor](/docs/transformers/main/en/model_doc/deit#transformers.DeiTImageProcessor) and [RobertaTokenizer](/docs/transformers/main/en/model_doc/longformer#transformers.RobertaTokenizer)/[XLMRobertaTokenizer](/docs/transformers/main/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) into a single instance of [TrOCRProcessor](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRProcessor) to handle images and text.
- TrOCR is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Resources

- A blog post on [Accelerating Document AI](https://huggingface.co/blog/document-ai) with TrOCR.
- A blog post on how to [Document AI](https://github.com/philschmid/document-ai-transformers) with TrOCR.
- A notebook on how to [finetune TrOCR on IAM Handwriting Database using Seq2SeqTrainer](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
- An interactive-demo on [TrOCR handwritten character recognition](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).
- A notebook on [inference with TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb) and Gradio demo.
- A notebook on [evaluating TrOCR on the IAM test set](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb).

## TrOCRConfig[[transformers.TrOCRConfig]]

#### transformers.TrOCRConfig[[transformers.TrOCRConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/configuration_trocr.py#L24)

This is the configuration class to store the configuration of a [TrOCRForCausalLM](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRForCausalLM). It is used to instantiate an
TrOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the TrOCR
[microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import TrOCRConfig, TrOCRForCausalLM

>>> # Initializing a TrOCR-base style configuration
>>> configuration = TrOCRConfig()

>>> # Initializing a model (with random weights) from the TrOCR-base style configuration
>>> model = TrOCRForCausalLM(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 50265) : Vocabulary size of the TrOCR model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [TrOCRForCausalLM](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRForCausalLM).

d_model (`int`, *optional*, defaults to 1024) : Dimensionality of the layers and the pooler layer.

decoder_layers (`int`, *optional*, defaults to 12) : Number of decoder layers.

decoder_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer decoder.

decoder_ffn_dim (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.

activation_function (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, and pooler.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

activation_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for activations inside the fully connected layer.

init_std (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

decoder_layerdrop (`float`, *optional*, defaults to 0.0) : The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models).

scale_embedding (`bool`, *optional*, defaults to `False`) : Whether or not to scale the word embeddings by sqrt(d_model).

use_learned_position_embeddings (`bool`, *optional*, defaults to `True`) : Whether or not to use learned position embeddings. If not, sinusoidal position embeddings will be used.

layernorm_embedding (`bool`, *optional*, defaults to `True`) : Whether or not to use a layernorm after the word + position embeddings.

## TrOCRProcessor[[transformers.TrOCRProcessor]]

#### transformers.TrOCRProcessor[[transformers.TrOCRProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/processing_trocr.py#L31)

Constructs a TrOCR processor which wraps a vision image processor and a TrOCR tokenizer into a single processor.

[TrOCRProcessor](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRProcessor) offers all the functionalities of [`ViTImageProcessor`/`DeiTImageProcessor`] and
[`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [__call__()](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for
more information.

__call__transformers.TrOCRProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/processing_trocr.py#L49[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "text", "val": ": typing.Union[str, list[str], list[list[str]], NoneType] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.trocr.processing_trocr.TrOCRProcessorKwargs]"}]

When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
`__call__()` and returns its output. If used in the context
`~TrOCRProcessor.as_target_processor` this method forwards all its arguments to TrOCRTokenizer's
`~TrOCRTokenizer.__call__`. Please refer to the docstring of the above two methods for more information.

**Parameters:**

image_processor ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*) : An instance of [`ViTImageProcessor`/`DeiTImageProcessor`]. The image processor is a required input.

tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*) : An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
#### from_pretrained[[transformers.TrOCRProcessor.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1349)

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
#### save_pretrained[[transformers.TrOCRProcessor.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L778)

Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
can be reloaded using the [from_pretrained()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### batch_decode[[transformers.TrOCRProcessor.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1520)

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.
#### decode[[transformers.TrOCRProcessor.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1529)

This method forwards all its arguments to PreTrainedTokenizer's [decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## TrOCRForCausalLM[[transformers.TrOCRForCausalLM]]

#### transformers.TrOCRForCausalLM[[transformers.TrOCRForCausalLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/modeling_trocr.py#L649)

The TrOCR Decoder with a language modeling head. Can be used as the decoder part of [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) and

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.TrOCRForCausalLM.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/modeling_trocr.py#L675[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **encoder_hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
- **encoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TrOCRConfig](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
The [TrOCRForCausalLM](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import (
...     TrOCRConfig,
...     TrOCRProcessor,
...     TrOCRForCausalLM,
...     ViTConfig,
...     ViTModel,
...     VisionEncoderDecoderModel,
... )
>>> import requests
>>> from PIL import Image

>>> # TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel
>>> # init vision2text model with random weights
>>> encoder = ViTModel(ViTConfig())
>>> decoder = TrOCRForCausalLM(TrOCRConfig())
>>> model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

>>> # If you want to start from the pretrained model, load the checkpoint with `VisionEncoderDecoderModel`
>>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
>>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

>>> # load image from the IAM dataset
>>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> pixel_values = processor(image, return_tensors="pt").pixel_values
>>> text = "industry, ' Mr. Brown commented icily. ' Let us have a"

>>> # training
>>> model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
>>> model.config.pad_token_id = processor.tokenizer.pad_token_id
>>> model.config.vocab_size = model.config.decoder.vocab_size

>>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
>>> outputs = model(pixel_values, labels=labels)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
5.30

>>> # inference
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> generated_text
'industry, " Mr. Brown commented icily. " Let us have a'
```

**Parameters:**

config ([TrOCRForCausalLM](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TrOCRConfig](/docs/transformers/main/en/model_doc/trocr#transformers.TrOCRConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
