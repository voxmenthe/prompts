*This model was released on 2023-10-17 and added to Hugging Face Transformers on 2023-10-19.*

# Fuyu

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Fuyu model was created by [ADEPT](https://www.adept.ai/blog/fuyu-8b), and authored by Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar.

The authors introduced Fuyu-8B, a decoder-only multimodal model based on the classic transformers architecture, with query and key normalization. A linear encoder is added to create multimodal embeddings from image inputs.

By treating image tokens like text tokens and using a special image-newline character, the model knows when an image line ends. Image positional embeddings are removed. This avoids the need for different training phases for various image resolutions. With 8 billion parameters and licensed under CC-BY-NC, Fuyu-8B is notable for its ability to handle both text and images, its impressive context size of 16K, and its overall performance.

The `Fuyu` models were trained using `bfloat16`, but the original inference uses `float16` The checkpoints uploaded on the hub use `dtype = 'float16'` which will be
used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`.

The `dtype` of the online weights is mostly irrelevant, unless you are using `dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online) then it will be cast to the default `dtype` of `torch` (becomes `torch.float32`). Users should specify the `dtype` they want, and if they don’t it will be `torch.float32`.

Finetuning the model in `float16` is not recommended and known to produce `nan`, as such the model should be fine-tuned in `bfloat16`.

Tips:

* To convert the model, you need to clone the original repository using `git clone https://github.com/persimmon-ai-labs/adept-inference`, then get the checkpoints:


```
git clone https://github.com/persimmon-ai-labs/adept-inference
wget path/to/fuyu-8b-model-weights.tar
tar -xvf fuyu-8b-model-weights.tar
python src/transformers/models/fuyu/convert_fuyu_weights_to_hf.py  --input_dir /path/to/downloaded/fuyu/weights/ --output_dir /output/path \
    --pt_model_path /path/to/fuyu_8b_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference
```

For the chat model:


```
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

Then, model can be loaded via:


```
from transformers import FuyuConfig, FuyuForCausalLM
model_config = FuyuConfig()
model = FuyuForCausalLM(model_config).from_pretrained('/output/path')
```

Inputs need to be passed through a specific Processor to have the correct formats.
A processor requires an image\_processor and a tokenizer. Hence, inputs can be loaded via:


```
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor


tokenizer = AutoTokenizer.from_pretrained('adept-hf-collab/fuyu-8b')
image_processor = FuyuImageProcessor()


processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
text_prompt = "Generate a coco-style caption.\\n"

bus_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
bus_image_pil = Image.open(io.BytesIO(requests.get(bus_image_url).content))
inputs_to_model = processor(images=bus_image_pil, text=text_prompt)
```

This model was contributed by [Molbap](https://huggingface.co/Molbap).
The original code can be found [here](https://github.com/persimmon-ai-labs/adept-inference).

* Fuyu uses a `sentencepiece` based tokenizer, with a `Unigram` model. It supports bytefallback, which is only available in `tokenizers==0.14.0` for the fast tokenizer.
  The `LlamaTokenizer` is used as it is a standard wrapper around sentencepiece.
* The authors suggest to use the following prompt for image captioning: `f"Generate a coco-style caption.\\n"`

## FuyuConfig

### class transformers.FuyuConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/configuration_fuyu.py#L25)

( vocab\_size = 262144 hidden\_size = 4096 intermediate\_size = 16384 num\_hidden\_layers = 36 num\_attention\_heads = 64 hidden\_act = 'relu2' max\_position\_embeddings = 16384 image\_size = 300 patch\_size = 30 num\_channels = 3 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 use\_cache = True tie\_word\_embeddings = False rope\_theta = 25000.0 rope\_scaling = None qk\_layernorm = True hidden\_dropout = 0.0 attention\_dropout = 0.0 partial\_rotary\_factor = 0.5 pad\_token\_id = None bos\_token\_id = 1 eos\_token\_id = 2 image\_token\_id = 71011 text\_config = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 262144) —
  Vocabulary size of the Fuyu model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 16384) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 36) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 64) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"relu2"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 16384) —
  The maximum sequence length that this model might ever be used with.
* **image\_size** (`int`, *optional*, defaults to 300) —
  The input image size.
* **patch\_size** (`int`, *optional*, defaults to 30) —
  The input vision transformer encoding patch size.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The input image number of channels.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`. Whether to tie weight embeddings
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie input and output embeddings.
* **rope\_theta** (`float`, *optional*, defaults to 25000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
  strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
  `{"type": strategy name, "factor": scaling factor}`. When using this flag, don’t update
  `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
  these scaling strategies behave:
  <https://www.reddit.com/r/LocalFuyu/comments/14mrgpr/dynamically_scaled_rope_further_increases/>. This is an
  experimental feature, subject to breaking API changes in future versions.
* **qk\_layernorm** (`bool`, *optional*, defaults to `True`) —
  Whether or not to normalize the Queries and Keys after projecting the hidden states
* **hidden\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio after applying the MLP to the hidden states.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio after computing the attention scores.
* **partial\_rotary\_factor** (`float`, *optional*, defaults to 0.5) —
  Percentage of the query and keys which will have rotary embedding.
* **pad\_token\_id** (`int`, *optional*) —
  The id of the *padding* token.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the *beginning-of-sequence* token.
* **eos\_token\_id** (`Union[int, list[int]]`, *optional*, defaults to 2) —
  The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
* **image\_token\_id** (`int`, *optional*, defaults to 71011) —
  The id of the image placeholder token.
* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize the ```` language```Aut ````.

This is the configuration class to store the configuration of a [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM). It is used to instantiate an
Fuyu model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
[adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import FuyuConfig

>>> # Initializing a Fuyu fuyu-7b style configuration
>>> configuration = FuyuConfig()
```

## FuyuModel

### class transformers.FuyuModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L64)

( config: FuyuConfig  )

Parameters

* **config** ([FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Fuyu model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L174)

( input\_ids: LongTensor = None image\_patches: Tensor = None image\_patches\_indices: Tensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **image\_patches** (`torch.FloatTensor` of shape `(batch_size, num_total_patches, patch_size_ x patch_size x num_channels)`, *optional*) —
  Image patches to be used as continuous embeddings. The patches are flattened and then projected to the
  hidden size of the model.
* **image\_patches\_indices** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Tensor of indices of the image patches in the input\_ids tensor.
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

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig)) and inputs.

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

The [FuyuModel](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### gather\_continuous\_embeddings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L92)

( word\_embeddings: Tensor continuous\_embeddings: list image\_patch\_input\_indices: Tensor  )

Parameters

* **word\_embeddings** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Tensor of word embeddings.
* **continuous\_embeddings** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`) —
  Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
  [num\_image\_embeddings, hidden], and num\_image\_embeddings needs to match the number of non-negative
  indices in image\_patch\_input\_indices for that batch element.
* **image\_patch\_input\_indices** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Tensor of indices of the image patches in the input\_ids tensor.

This function places the continuous\_embeddings into the word\_embeddings at the locations
indicated by image\_patch\_input\_indices. Different batch elements can have different numbers of continuous
embeddings.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L136)

( pixel\_values: FloatTensor \*\*kwargs  )

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images.

Encodes images into continuous embeddings that can be forwarded to the language model.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L150)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## FuyuForCausalLM

### class transformers.FuyuForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L253)

( config: FuyuConfig  )

Parameters

* **config** ([FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/modeling_fuyu.py#L279)

( input\_ids: LongTensor = None image\_patches: Tensor = None image\_patches\_indices: Tensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None logits\_to\_keep: typing.Optional[int] = 0 \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **image\_patches** (`torch.FloatTensor` of shape `(batch_size, num_total_patches, patch_size_ x patch_size x num_channels)`, *optional*) —
  Image patches to be used as continuous embeddings. The patches are flattened and then projected to the
  hidden size of the model.
* **image\_patches\_indices** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Tensor of indices of the image patches in the input\_ids tensor.
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **logits\_to\_keep** (`int`, *optional*, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig)) and inputs.

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

The [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import FuyuProcessor, FuyuForCausalLM
>>> from PIL import Image
>>> import requests

>>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
>>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

>>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> prompt = "Generate a coco-style caption.\n"

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")
>>> outputs = model(**inputs)

>>> generated_ids = model.generate(**inputs, max_new_tokens=7)
>>> generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)
>>> print(generation_text[0])
A blue bus parked on the side of a road.
```

## FuyuImageProcessor

### class transformers.FuyuImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/image_processing_fuyu.py#L182)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_pad: bool = True padding\_value: float = 1.0 padding\_mode: str = 'constant' do\_normalize: bool = True image\_mean: typing.Union[float, list[float]] = 0.5 image\_std: typing.Union[float, list[float]] = 0.5 do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 patch\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image to `size`.
* **size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 1080, "width": 1920}`):
  Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image to `size`.
* **padding\_value** (`float`, *optional*, defaults to 1.0) —
  The value to pad the image with.
* **padding\_mode** (`str`, *optional*, defaults to `"constant"`) —
  The padding mode to use when padding the image.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image.
* **image\_mean** (`float`, *optional*, defaults to 0.5) —
  The mean to use when normalizing the image.
* **image\_std** (`float`, *optional*, defaults to 0.5) —
  The standard deviation to use when normalizing the image.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `1 / 255`) —
  The factor to use when rescaling the image.
* **patch\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 30, "width": 30}`):
  Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.

This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
handle:

* Processing Images:
  Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
  dimensions. The image output is always img\_h, img\_w of (1080, 1920)

  Then, it patches up these images using the patchify\_image function.
* Creating Image Input IDs:
  For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
  variable-sized images, each line of patches is terminated with a newline ID.
* Image Patch Indices:
  For each image patch, the code maintains an index where these patches should be inserted in a token stream.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## FuyuProcessor

### class transformers.FuyuProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/processing_fuyu.py#L337)

( image\_processor tokenizer \*\*kwargs  )

Parameters

* **image\_processor** ([FuyuImageProcessor](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor)) —
  The image processor is a required input.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast)) —
  The tokenizer is a required input.

Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

[FuyuProcessor](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuProcessor) offers all the functionalities of [FuyuImageProcessor](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See the
[**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fuyu/processing_fuyu.py#L486)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None text: typing.Union[str, list[str], NoneType] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.fuyu.processing\_fuyu.FuyuProcessorKwargs]  ) → `FuyuBatchEncoding`

Parameters

* **images** (`PIL.Image.Image`, `list[PIL.Image.Image]`) —
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. Both channels-first and channels-last formats are supported.
* **text** (`str`, `list[str]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

Returns

`FuyuBatchEncoding`

A `FuyuBatchEncoding` with the following fields:

* **input\_ids** — Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
* **image\_patches** — List of Tensor of image patches. Returned when `images` is not `None`.
* **image\_patches\_indices** — Tensor of indices where patch embeddings have to be inserted by the model.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model when
  `return_attention_mask=True`.

Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
and `kwargs` arguments to LlamaTokenizerFast’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None` to
encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
FuyuImageProcessor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) if `images` is not `None`. Please refer to the docstring
of the above two methods for more information.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/fuyu.md)
