*This model was released on 2024-06-27 and added to Hugging Face Transformers on 2024-12-17.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# ColPali

[ColPali](https://huggingface.co/papers/2407.01449) is a model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColPali treats each page as an image. It uses [Paligemma-3B](./paligemma) to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.

This model was contributed by [@tonywu71](https://huggingface.co/tonywu71) (ILLUIN Technology) and [@yonigozlan](https://huggingface.co/yonigozlan) (HuggingFace).

You can find all the original ColPali checkpoints under Vidore’s [Hf-native ColVision Models](https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7) collection.

Click on the ColPali models in the right sidebar for more examples of how to use ColPali for image retrieval.

image retrieval


```
import requests
import torch
from PIL import Image

from transformers import ColPaliForRetrieval, ColPaliProcessor


# Load the model and the processor
model_name = "vidore/colpali-v1.3-hf"

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",  # "cpu", "cuda", "xpu", or "mps" for Apple Silicon
)
processor = ColPaliProcessor.from_pretrained(model_name)

# The document page screenshots from your corpus
url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

# The queries you want to retrieve documents for
queries = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]

# Process the inputs
inputs_images = processor(images=images).to(model.device)
inputs_text = processor(text=queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

If you have issue with loading the images with PIL, you can use the following code to create dummy images:


```
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to int4.


```
import requests
import torch
from PIL import Image

from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor


model_name = "vidore/colpali-v1.3-hf"

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

processor = ColPaliProcessor.from_pretrained(model_name)

url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

queries = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]

# Process the inputs
inputs_images = processor(images=images, return_tensors="pt").to(model.device)
inputs_text = processor(text=queries, return_tensors="pt").to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

## Notes

* [score\_retrieval()](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor.score_retrieval) returns a 2D tensor where the first dimension is the number of queries and the second dimension is the number of images. A higher score indicates more similarity between the query and image.

## ColPaliConfig

### class transformers.ColPaliConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/configuration_colpali.py#L27)

( vlm\_config = None text\_config = None embedding\_dim: int = 128 \*\*kwargs  )

Parameters

* **vlm\_config** (`PretrainedConfig`, *optional*) —
  Configuration of the VLM backbone model.
* **text\_config** (`PretrainedConfig`, *optional*) —
  Configuration of the text backbone model. Overrides the `text_config` attribute of the `vlm_config` if provided.
* **embedding\_dim** (`int`, *optional*, defaults to 128) —
  Dimension of the multi-vector embeddings produced by the model.

Configuration class to store the configuration of a [ColPaliForRetrieval](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliForRetrieval). It is used to instantiate an instance
of `ColPaliForRetrieval` according to the specified arguments, defining the model architecture following the methodology
from the “ColPali: Efficient Document Retrieval with Vision Language Models” paper.

Creating a configuration with the default settings will result in a configuration where the VLM backbone is set to the
default PaliGemma configuration, i.e the one from [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2).

Note that contrarily to what the class name suggests (actually the name refers to the ColPali **methodology**), you can
use a different VLM backbone model than PaliGemma by passing the corresponding VLM configuration to the class constructor.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
from transformers.models.colpali import ColPaliConfig, ColPaliForRetrieval

config = ColPaliConfig()
model = ColPaliForRetrieval(config)
```

## ColPaliProcessor

### class transformers.ColPaliProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L75)

( image\_processor = None tokenizer = None chat\_template = None visual\_prompt\_prefix: str = 'Describe the image.' query\_prefix: str = 'Question: '  )

Parameters

* **image\_processor** ([SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast), *optional*) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.
* **visual\_prompt\_prefix** (`str`, *optional*, defaults to `"Describe the image."`) —
  A string that gets tokenized and prepended to the image tokens.
* **query\_prefix** (`str`, *optional*, defaults to `"Question -- "`):
  A prefix to be used for the query.

Constructs a ColPali processor which wraps a PaliGemmaProcessor and special methods to process images and queries, as
well as to compute the late-interaction retrieval score.

[ColPaliProcessor](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor) offers all the functionalities of [PaliGemmaProcessor](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor). See the `__call__()`
for more information.

#### process\_images

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L280)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.colpali.processing\_colpali.ColPaliProcessorKwargs]  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
  number of channels, H and W are image height and width.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names` and if `text` is not
  `None`).
* **pixel\_values** — Pixel values to be fed to a model. Returned when `images` is not `None`.

Prepare for the model one or several image(s). This method is a wrapper around the `__call__` method of the ColPaliProcessor’s
`ColPaliProcessor.__call__()`.

This method forwards the `images` and `kwargs` arguments to the image processor.

#### process\_queries

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L315)

( text: typing.Union[str, list[str]] \*\*kwargs: typing\_extensions.Unpack[transformers.models.colpali.processing\_colpali.ColPaliProcessorKwargs]  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names` and if `text` is not
  `None`).

Prepare for the model one or several texts. This method is a wrapper around the `__call__` method of the ColPaliProcessor’s
`ColPaliProcessor.__call__()`.

This method forwards the `text` and `kwargs` arguments to the tokenizer.

#### score\_retrieval

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L349)

( query\_embeddings: typing.Union[ForwardRef('torch.Tensor'), list['torch.Tensor']] passage\_embeddings: typing.Union[ForwardRef('torch.Tensor'), list['torch.Tensor']] batch\_size: int = 128 output\_dtype: typing.Optional[ForwardRef('torch.dtype')] = None output\_device: typing.Union[ForwardRef('torch.device'), str] = 'cpu'  ) → `torch.Tensor`

Parameters

* **query\_embeddings** (`Union[torch.Tensor, list[torch.Tensor]`) — Query embeddings.
* **passage\_embeddings** (`Union[torch.Tensor, list[torch.Tensor]`) — Passage embeddings.
* **batch\_size** (`int`, *optional*, defaults to 128) — Batch size for computing scores.
* **output\_dtype** (`torch.dtype`, *optional*, defaults to `torch.float32`) — The dtype of the output tensor.
  If `None`, the dtype of the input embeddings is used.
* **output\_device** (`torch.device` or `str`, *optional*, defaults to “cpu”) — The device of the output tensor.

Returns

`torch.Tensor`

A tensor of shape `(n_queries, n_passages)` containing the scores. The score
tensor is saved on the “cpu” device.

Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
image of a document page.

Because the embedding tensors are multi-vector and can thus have different shapes, they
should be fed as:
(1) a list of tensors, where the i-th tensor is of shape (sequence\_length\_i, embedding\_dim)
(2) a single tensor of shape (n\_passages, max\_sequence\_length, embedding\_dim) -> usually
obtained by padding the list of tensors.

## ColPaliForRetrieval

### class transformers.ColPaliForRetrieval

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/modeling_colpali.py#L102)

( config: ColPaliConfig  )

Parameters

* **config** ([ColPaliConfig](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ColPali architecture leverages VLMs to construct efficient multi-vector embeddings directly
from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
between these document embeddings and the corresponding query embeddings, using the late interaction method
introduced in ColBERT.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a
single model that can take into account both the textual and visual content (layout, charts, etc.) of a document.

ColPali is part of the ColVision model family, which was first introduced in the following paper:
[*ColPali: Efficient Document Retrieval with Vision Language Models*](https://huggingface.co/papers/2407.01449).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/modeling_colpali.py#L126)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.colpali.modeling_colpali.ColPaliForRetrievalOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([ColPaliProcessor](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor) uses
  `image_processor_class` for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.colpali.modeling_colpali.ColPaliForRetrievalOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.colpali.modeling_colpali.ColPaliForRetrievalOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ColPaliConfig](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **embeddings** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — The embeddings of the model.
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

The [ColPaliForRetrieval](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliForRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/colpali.md)
