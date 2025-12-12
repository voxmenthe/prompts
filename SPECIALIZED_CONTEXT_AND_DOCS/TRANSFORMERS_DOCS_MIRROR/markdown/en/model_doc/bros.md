*This model was released on 2021-08-10 and added to Hugging Face Transformers on 2023-09-15.*

# BROS

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The BROS model was proposed in [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://huggingface.co/papers/2108.04539) by Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park.

BROS stands for *BERT Relying On Spatiality*. It is an encoder-only Transformer model that takes a sequence of tokens and their bounding boxes as inputs and outputs a sequence of hidden states. BROS encode relative spatial information instead of using absolute spatial information.

It is pre-trained with two objectives: a token-masked language modeling objective (TMLM) used in BERT, and a novel area-masked language modeling objective (AMLM)
In TMLM, tokens are randomly masked, and the model predicts the masked tokens using spatial information and other unmasked tokens.
AMLM is a 2D version of TMLM. It randomly masks text tokens and predicts with the same information as TMLM, but it masks text blocks (areas).

`BrosForTokenClassification` has a simple linear layer on top of BrosModel. It predicts the label of each token.
`BrosSpadeEEForTokenClassification` has an `initial_token_classifier` and `subsequent_token_classifier` on top of BrosModel. `initial_token_classifier` is used to predict the first token of each entity, and `subsequent_token_classifier` is used to predict the next token of within entity. `BrosSpadeELForTokenClassification` has an `entity_linker` on top of BrosModel. `entity_linker` is used to predict the relation between two entities.

`BrosForTokenClassification` and `BrosSpadeEEForTokenClassification` essentially perform the same job. However, `BrosForTokenClassification` assumes input tokens are perfectly serialized (which is very challenging task since they exist in a 2D space), while `BrosSpadeEEForTokenClassification` allows for more flexibility in handling serialization errors as it predicts next connection tokens from one token.

`BrosSpadeELForTokenClassification` perform the intra-entity linking task. It predicts relation from one token (of one entity) to another token (of another entity) if these two entities share some relation.

BROS achieves comparable or better result on Key Information Extraction (KIE) benchmarks such as FUNSD, SROIE, CORD and SciTSR, without relying on explicit visual features.

The abstract from the paper is the following:

*Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space. Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout. On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. Specifically, we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy. With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE*, CORD, and SciTSR) without relying on visual features. This paper also reveals two real-world challenges in KIE tasks-(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples-and demonstrates the superiority of BROS over previous methods.\*

This model was contributed by [jinho8345](https://huggingface.co/jinho8345). The original code can be found [here](https://github.com/clovaai/bros).

## Usage tips and examples

* [forward()](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel.forward) requires `input_ids` and `bbox` (bounding box). Each bounding box should be in (x0, y0, x1, y1) format (top-left corner, bottom-right corner). Obtaining of Bounding boxes depends on external OCR system. The `x` coordinate should be normalized by document image width, and the `y` coordinate should be normalized by document image height.


```
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
    # here, bboxes are numpy array

    # Normalize bbox -> 0 ~ 1
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height
```

* [`~transformers.BrosForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`] require not only `input_ids` and `bbox` but also `box_first_token_mask` for loss calculation. It is a mask to filter out non-first tokens of each box. You can obtain this mask by saving start token indices of bounding boxes when creating `input_ids` from words. You can make `box_first_token_mask` with following code,


```
def make_box_first_token_mask(bboxes, words, tokenizer, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    # encode(tokenize) each word from words (list[str])
    input_ids_list: list[list[int]] = [tokenizer.encode(e, add_special_tokens=False) for e in words]

    # get the length of each box
    tokens_length_list: list[int] = [len(l) for l in input_ids_list]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    # filter out the indices that are out of max_seq_length
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    # set box_start_token_indices to True
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask
```

## Resources

* Demo scripts can be found [here](https://github.com/clovaai/bros).

## BrosConfig

### class transformers.BrosConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/configuration_bros.py#L24)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 dim\_bbox = 8 bbox\_scale = 100.0 n\_relations = 1 classifier\_dropout\_prob = 0.1 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Bros model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel) or `TFBrosModel`.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel) or `TFBrosModel`.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The index of the padding token in the token vocabulary.
* **dim\_bbox** (`int`, *optional*, defaults to 8) —
  The dimension of the bounding box coordinates. (x0, y1, x1, y0, x1, y1, x0, y1)
* **bbox\_scale** (`float`, *optional*, defaults to 100.0) —
  The scale factor of the bounding box coordinates.
* **n\_relations** (`int`, *optional*, defaults to 1) —
  The number of relations for SpadeEE(entity extraction), SpadeEL(entity linking) head.
* **classifier\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the classifier head.

This is the configuration class to store the configuration of a [BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel) or a `TFBrosModel`. It is used to
instantiate a Bros model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Bros
[jinho8345/bros-base-uncased](https://huggingface.co/jinho8345/bros-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import BrosConfig, BrosModel

>>> # Initializing a BROS jinho8345/bros-base-uncased style configuration
>>> configuration = BrosConfig()

>>> # Initializing a model from the jinho8345/bros-base-uncased style configuration
>>> model = BrosModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## BrosProcessor

### class transformers.BrosProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/processing_bros.py#L26)

( tokenizer = None \*\*kwargs  )

Parameters

* **tokenizer** (`BertTokenizerFast`, *optional*) —
  An instance of [‘BertTokenizerFast`]. The tokenizer is a required input.

Constructs a Bros processor which wraps a BERT tokenizer.

[BrosProcessor](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosProcessor) offers all the functionalities of [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast). See the docstring of
[**call**()](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/processing_bros.py#L47)

( text: typing.Union[str, list[str], list[list[str]]] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  )

This method uses [BertTokenizerFast.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) to prepare text for the model.

Please refer to the docstring of the above two methods for more information.

## BrosModel

### class transformers.BrosModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L599)

( config add\_pooling\_layer = True  )

Parameters

* **config** ([BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Bros Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L630)

( input\_ids: typing.Optional[torch.Tensor] = None bbox: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`'torch.FloatTensor'` of shape ‘(batch\_size, num\_boxes, 4)’) —
  Bounding box coordinates for each token in the input sequence. Each bounding box is a list of four values
  (x1, y1, x2, y2), where (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner of the
  bounding box.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_hidden\_states** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

The [BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import BrosProcessor, BrosModel

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosModel.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
>>> last_hidden_states = outputs.last_hidden_state
```

## BrosForTokenClassification

### class transformers.BrosForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L761)

( config  )

Parameters

* **config** ([BrosForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Bros transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L777)

( input\_ids: typing.Optional[torch.Tensor] = None bbox: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None bbox\_first\_token\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`'torch.FloatTensor'` of shape ‘(batch\_size, num\_boxes, 4)’) —
  Bounding box coordinates for each token in the input sequence. Each bounding box is a list of four values
  (x1, y1, x2, y2), where (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner of the
  bounding box.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **bbox\_first\_token\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the first token of each bounding box. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BrosForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import BrosProcessor, BrosForTokenClassification

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
```

## BrosSpadeEEForTokenClassification

### class transformers.BrosSpadeEEForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L870)

( config  )

Parameters

* **config** ([BrosSpadeEEForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosSpadeEEForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bros Model with a token classification head on top (initial\_token\_layers and subsequent\_token\_layer on top of the
hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. The initial\_token\_classifier is used to
predict the first token of each entity, and the subsequent\_token\_classifier is used to predict the subsequent
tokens within an entity. Compared to BrosForTokenClassification, this model is more robust to serialization errors
since it predicts next token from one token.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L898)

( input\_ids: typing.Optional[torch.Tensor] = None bbox: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None bbox\_first\_token\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None initial\_token\_labels: typing.Optional[torch.Tensor] = None subsequent\_token\_labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.bros.modeling_bros.BrosSpadeOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`'torch.FloatTensor'` of shape ‘(batch\_size, num\_boxes, 4)’) —
  Bounding box coordinates for each token in the input sequence. Each bounding box is a list of four values
  (x1, y1, x2, y2), where (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner of the
  bounding box.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **bbox\_first\_token\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the first token of each bounding box. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **initial\_token\_labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for the initial token classification.
* **subsequent\_token\_labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for the subsequent token classification.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.bros.modeling_bros.BrosSpadeOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.bros.modeling_bros.BrosSpadeOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **initial\_token\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores for entity initial tokens (before SoftMax).
* **subsequent\_token\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length+1)`) — Classification scores for entity sequence tokens (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BrosSpadeEEForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosSpadeEEForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import BrosProcessor, BrosSpadeEEForTokenClassification

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
```

## BrosSpadeELForTokenClassification

### class transformers.BrosSpadeELForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L1020)

( config  )

Parameters

* **config** ([BrosSpadeELForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosSpadeELForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bros Model with a token classification head on top (a entity\_linker layer on top of the hidden-states output) e.g.
for Entity-Linking. The entity\_linker is used to predict intra-entity links (one entity to another entity).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bros/modeling_bros.py#L1037)

( input\_ids: typing.Optional[torch.Tensor] = None bbox: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None bbox\_first\_token\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`'torch.FloatTensor'` of shape ‘(batch\_size, num\_boxes, 4)’) —
  Bounding box coordinates for each token in the input sequence. Each bounding box is a list of four values
  (x1, y1, x2, y2), where (x1, y1) is the top left corner, and (x2, y2) is the bottom right corner of the
  bounding box.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **bbox\_first\_token\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to indicate the first token of each bounding box. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BrosSpadeELForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosSpadeELForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bros.md)
