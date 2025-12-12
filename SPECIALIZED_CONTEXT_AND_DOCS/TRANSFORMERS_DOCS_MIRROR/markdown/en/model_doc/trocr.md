*This model was released on 2021-09-21 and added to Hugging Face Transformers on 2021-10-13.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# TrOCR

[TrOCR](https://huggingface.co/papers/2109.10282) is a text recognition model for both image understanding and text generation. It doesn’t require separate models for image processing or character generation. TrOCR is a simple single end-to-end system that uses a transformer to handle visual understanding and text generation.

You can find all the original TrOCR checkpoints under the [Microsoft](https://huggingface.co/microsoft/models?search=trocr) organization.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trocr_architecture.jpg) TrOCR architecture. Taken from the [original paper](https://huggingface.co/papers/2109.10282).

This model was contributed by [nielsr](https://huggingface.co/nielsr).

Click on the TrOCR models in the right sidebar for more examples of how to apply TrOCR to different image and text tasks.

The example below demonstrates how to perform optical character recognition (OCR) with the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

AutoModel


```
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


```
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

* TrOCR wraps [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor)/[DeiTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTImageProcessor) and [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer)/[XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) into a single instance of [TrOCRProcessor](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRProcessor) to handle images and text.
* TrOCR is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Resources

* A blog post on [Accelerating Document AI](https://huggingface.co/blog/document-ai) with TrOCR.
* A blog post on how to [Document AI](https://github.com/philschmid/document-ai-transformers) with TrOCR.
* A notebook on how to [finetune TrOCR on IAM Handwriting Database using Seq2SeqTrainer](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
* An interactive-demo on [TrOCR handwritten character recognition](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).
* A notebook on [inference with TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb) and Gradio demo.
* A notebook on [evaluating TrOCR on the IAM test set](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb).

## TrOCRConfig

### class transformers.TrOCRConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/trocr/configuration_trocr.py#L24)

( vocab\_size = 50265 d\_model = 1024 decoder\_layers = 12 decoder\_attention\_heads = 16 decoder\_ffn\_dim = 4096 activation\_function = 'gelu' max\_position\_embeddings = 512 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 decoder\_start\_token\_id = 2 init\_std = 0.02 decoder\_layerdrop = 0.0 use\_cache = True scale\_embedding = False use\_learned\_position\_embeddings = True layernorm\_embedding = True pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 50265) —
  Vocabulary size of the TrOCR model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [TrOCRForCausalLM](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRForCausalLM).
* **d\_model** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the layers and the pooler layer.
* **decoder\_layers** (`int`, *optional*, defaults to 12) —
  Number of decoder layers.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the pooler. If string, `"gelu"`, `"relu"`,
  `"silu"` and `"gelu_new"` are supported.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the decoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **scale\_embedding** (`bool`, *optional*, defaults to `False`) —
  Whether or not to scale the word embeddings by sqrt(d\_model).
* **use\_learned\_position\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use learned position embeddings. If not, sinusoidal position embeddings will be used.
* **layernorm\_embedding** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use a layernorm after the word + position embeddings.

This is the configuration class to store the configuration of a [TrOCRForCausalLM](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRForCausalLM). It is used to instantiate an
TrOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the TrOCR
[microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import TrOCRConfig, TrOCRForCausalLM

>>> # Initializing a TrOCR-base style configuration
>>> configuration = TrOCRConfig()

>>> # Initializing a model (with random weights) from the TrOCR-base style configuration
>>> model = TrOCRForCausalLM(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## TrOCRProcessor

### class transformers.TrOCRProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/trocr/processing_trocr.py#L33)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*) —
  An instance of [`ViTImageProcessor`/`DeiTImageProcessor`]. The image processor is a required input.
* **tokenizer** ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*) —
  An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.

Constructs a TrOCR processor which wraps a vision image processor and a TrOCR tokenizer into a single processor.

[TrOCRProcessor](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRProcessor) offers all the functionalities of [`ViTImageProcessor`/`DeiTImageProcessor`] and
[`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [**call**()](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for
more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/trocr/processing_trocr.py#L72)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None text: typing.Union[str, list[str], list[list[str]]] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.trocr.processing\_trocr.TrOCRProcessorKwargs]  )

When used in normal mode, this method forwards all its arguments to AutoImageProcessor’s
`__call__()` and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to TrOCRTokenizer’s
`~TrOCRTokenizer.__call__`. Please refer to the docstring of the above two methods for more information.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1272)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* \***\*kwargs** —
  Additional keyword arguments passed along to both
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and
  `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L653)

( save\_directory push\_to\_hub: bool = False legacy\_serialization: bool = True \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
  be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **legacy\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to save processor attributes in separate config files (legacy) or in processor’s config
  file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
  Set to `legacy_serialization=True` until then.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it
can be reloaded using the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## TrOCRForCausalLM

### class transformers.TrOCRForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/trocr/modeling_trocr.py#L711)

( config  )

Parameters

* **config** ([TrOCRForCausalLM](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The TrOCR Decoder with a language modeling head. Can be used as the decoder part of [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) and

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/trocr/modeling_trocr.py#L743)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

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
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
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
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TrOCRConfig](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.

The [TrOCRForCausalLM](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/trocr.md)
