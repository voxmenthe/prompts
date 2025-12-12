*This model was released on 2023-02-07 and added to Hugging Face Transformers on 2023-06-20.*

# GPTSAN-japanese

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The [GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) model was released in the repository by Toshiyuki Sakamoto (tanreinama).

GPTSAN is a Japanese language model using Switch Transformer. It has the same structure as the model introduced as Prefix LM
in the T5 paper, and support both Text Generation and Masked Language Modeling tasks. These basic tasks similarly can
fine-tune for translation or summarization.

### Usage example

The `generate()` method can be used to generate text using GPTSAN-Japanese model.


```
>>> from transformers import AutoModel, AutoTokenizer, infer_device
>>> import torch

>>> device = infer_device()
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> x_tok = tokenizer("は、", prefix_text="織田信長", return_tensors="pt")
>>> torch.manual_seed(0)
>>> gen_tok = model.generate(x_tok.input_ids.to(model.device), token_type_ids=x_tok.token_type_ids.to(mdoel.device), max_new_tokens=20)
>>> tokenizer.decode(gen_tok[0])
'織田信長は、2004年に『戦国BASARA』のために、豊臣秀吉'
```

## GPTSAN Features

GPTSAN has some unique features. It has a model structure of Prefix-LM. It works as a shifted Masked Language Model for Prefix Input tokens. Un-prefixed inputs behave like normal generative models.
The Spout vector is a GPTSAN specific input. Spout is pre-trained with random inputs, but you can specify a class of text or an arbitrary vector during fine-tuning. This allows you to indicate the tendency of the generated text.
GPTSAN has a sparse Feed Forward based on Switch-Transformer. You can also add other layers and train them partially. See the original GPTSAN repository for details.

### Prefix-LM Model

GPTSAN has the structure of the model named Prefix-LM in the `T5` paper. (The original GPTSAN repository calls it `hybrid`)
In GPTSAN, the `Prefix` part of Prefix-LM, that is, the input position that can be referenced by both tokens, can be specified with any length.
Arbitrary lengths can also be specified differently for each batch.
This length applies to the text entered in `prefix_text` for the tokenizer.
The tokenizer returns the mask of the `Prefix` part of Prefix-LM as `token_type_ids`.
The model treats the part where `token_type_ids` is 1 as a `Prefix` part, that is, the input can refer to both tokens before and after.

## Usage tips

Specifying the Prefix part is done with a mask passed to self-attention.
When token\_type\_ids=None or all zero, it is equivalent to regular causal mask

for example:

> > > x\_token = tokenizer(“ｱｲｳｴ”)
> > > input\_ids: | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
> > > token\_type\_ids: | 1 | 0 | 0 | 0 | 0 | 0 |
> > > prefix\_lm\_mask:
> > > SOT | 1 0 0 0 0 0 |
> > > SEG | 1 1 0 0 0 0 |
> > > ｱ | 1 1 1 0 0 0 |
> > > ｲ | 1 1 1 1 0 0 |
> > > ｳ | 1 1 1 1 1 0 |
> > > ｴ | 1 1 1 1 1 1 |

> > > x\_token = tokenizer("", prefix\_text=“ｱｲｳｴ”)
> > > input\_ids: | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
> > > token\_type\_ids: | 1 | 1 | 1 | 1 | 1 | 0 |
> > > prefix\_lm\_mask:
> > > SOT | 1 1 1 1 1 0 |
> > > ｱ | 1 1 1 1 1 0 |
> > > ｲ | 1 1 1 1 1 0 |
> > > ｳ | 1 1 1 1 1 0 |
> > > ｴ | 1 1 1 1 1 0 |
> > > SEG | 1 1 1 1 1 1 |

> > > x\_token = tokenizer(“ｳｴ”, prefix\_text=“ｱｲ”)
> > > input\_ids: | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
> > > token\_type\_ids: | 1 | 1 | 1 | 0 | 0 | 0 |
> > > prefix\_lm\_mask:
> > > SOT | 1 1 1 0 0 0 |
> > > ｱ | 1 1 1 0 0 0 |
> > > ｲ | 1 1 1 0 0 0 |
> > > SEG | 1 1 1 1 0 0 |
> > > ｳ | 1 1 1 1 1 0 |
> > > ｴ | 1 1 1 1 1 1 |

### Spout Vector

A Spout Vector is a special vector for controlling text generation.
This vector is treated as the first embedding in self-attention to bring extraneous attention to the generated tokens.
In the pre-trained model published from `Tanrei/GPTSAN-japanese`, the Spout Vector is a 128-dimensional vector that passes through 8 fully connected layers in the model and is projected into the space acting as external attention.
The Spout Vector projected by the fully connected layer is split to be passed to all self-attentions.

## GPTSanJapaneseConfig

### class transformers.GPTSanJapaneseConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/configuration_gptsan_japanese.py#L24)

( vocab\_size = 36000 max\_position\_embeddings = 1280 d\_model = 1024 d\_ff = 8192 d\_ext = 4096 d\_spout = 128 num\_switch\_layers = 10 num\_ext\_layers = 0 num\_heads = 16 num\_experts = 16 expert\_capacity = 128 dropout\_rate = 0.0 layer\_norm\_epsilon = 1e-05 router\_bias = False router\_jitter\_noise = 0.0 router\_dtype = 'float32' router\_ignore\_padding\_tokens = False output\_hidden\_states = False output\_attentions = False initializer\_factor = 0.002 output\_router\_logits = False use\_cache = True separator\_token\_id = 35998 pad\_token\_id = 35995 eos\_token\_id = 35999 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 36000) —
  Vocabulary size of the GPTSANJapanese model. Defines the number of different tokens that can be represented
  by the `inputs_ids` passed when calling [GPTSanJapaneseModel](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseModel).
* **max\_position\_embeddings** (`int`, *optional*, defaults to 1280) —
  The maximum sequence length that this model might ever be used with. Defaults set this to 1280.
* **d\_model** (`int`, *optional*, defaults to 1024) —
  Size of the encoder layers and the pooler layer.
* **d\_ff** (`int`, *optional*, defaults to 8192) —
  Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
* **d\_ext** (`int`, *optional*, defaults to 4096) —
  Size of the intermediate feed forward layer in each Extra-layers.
* **d\_spout** (`int`, *optional*, defaults to 128) —
  Size of the `spout` vector.
* **num\_switch\_layers** (`int`, *optional*, defaults to 10) —
  Number of layers in the Switch Transformer layer.
* **num\_ext\_layers** (`int`, *optional*, defaults to 0) —
  Number of layers in the Extra-layers.
* **num\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_experts** (`int`, *optional*, defaults to 16) —
  Number of experts for each SwitchTransformer layer.
* **expert\_capacity** (`int`, *optional*, defaults to 128) —
  Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
  Transformer.
* **dropout\_rate** (`float`, *optional*, defaults to 0.0) —
  The ratio for all dropout layers.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-5) —
  The epsilon used by the layer normalization layers.
* **router\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the router.
* **router\_jitter\_noise** (`float`, *optional*, defaults to 0.0) —
  Amount of noise to add to the router. Set it to 0.0 during prediction or set small value (usually 1e-2)
  during training.
* **router\_dtype** (`str`, *optional*, default to `"float32"`) —
  The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
  *selective precision* discussion in [the paper](https://huggingface.co/papers/2101.03961).
* **router\_ignore\_padding\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether to ignore padding tokens when routing.
* **output\_hidden\_states** (`bool`, *optional*, default to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the attentions tensors of all attention layers.
* **initializer\_factor** (`float`, *optional*, defaults to 0.002) —
  A factor for initializing all weight matrices.
* **output\_router\_logits** (`bool`, *optional*, default to `False`) —
  Whether or not to return the router logits of all experts.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models)

This is the configuration class to store the configuration of a [GPTSanJapaneseModel](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseModel). It is used to instantiate
a GPTSANJapanese model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the GPTSANJapanese
[Tanrei/GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## GPTSanJapaneseTokenizer

### class transformers.GPTSanJapaneseTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/tokenization_gptsan_japanese.py#L63)

( vocab\_file emoji\_file unk\_token = '<|nottoken|>' pad\_token = '<|separator|>' bos\_token = '<|startoftext|>' eos\_token = '<|endoftext|>' sep\_token = '<|segmenter|>' do\_clean\_text = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **emoji\_file** (`str`) —
  File containing the emoji.
* **unk\_token** (`str`, *optional*, defaults to `"<|nottoken|>"`) —
  The token used for unknown character
* **pad\_token** (`str`, *optional*, defaults to `"<|separator|>"`) —
  The token used for padding
* **bos\_token** (`str`, *optional*, defaults to `"<|startoftext|>"`) —
  The beginning of sequence token.
* **eos\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The end of sequence token.
* **sep\_token** (`str`, *optional*, defaults to `"<|segmenter|>"`) —
  A special token to separate token to prefix part and general input part.
* **do\_clean\_text** (`bool`, *optional*, defaults to `False`) —
  Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.

This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications

* Decoding byte0~byte255 tokens correctly
* Added bagofword token handling
* Return token\_type\_ids for Prefix-LM model
  The bagofword token represents a repetition of the previous token and is converted to 3 consecutive tokens when
  decoding In addition, the original Japanese special Sub-Word-Encoding has been released in this repository
  (<https://github.com/tanreinama/Japanese-BPEEncoder_V2>). The token\_type\_ids is a mask indicating the prefix input
  position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix\_text, or specify a
  sentence of the prefix part and the part after it as a text pair of batch input.

Example:


```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> # You can confirm both 慶応 and 慶應 are encoded to 17750
>>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
[35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

>>> # Both 慶応 and 慶應 are decoded to 慶応
>>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
'吾輩は猫である🐯。実は慶応(慶応)大学出身'
```

Example for Prefix-LM:


```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
[35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

>>> # Mask for Prefix-LM inputs
>>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Example for batch encode:


```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
[[35993, 35998, 8640, 25948, 35993, 35998, 30647, 35675, 35999, 35999], [35993, 35998, 10382, 9868, 35993, 35998, 30646, 9459, 30646, 35675]]

>>> # Mask for Prefix-LM inputs
>>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

>>> # Mask for padding
>>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/tokenization_gptsan_japanese.py#L201)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/tokenization_gptsan_japanese.py#L270)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  )

The tokenizer returns token\_type\_ids as separators between the Prefix part and the rest.
token\_type\_ids is 1 for the Prefix part and 0 for the rest of the token.

Example:


```
>>> from transformers import GPTSanJapaneseTokenizer

>>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> x_token = tokenizer("ｱｲｳｴ")
>>> # input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
>>> # token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |

>>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
>>> # input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
>>> # token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |

>>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
>>> # input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
>>> # token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
```

## GPTSanJapaneseModel

### class transformers.GPTSanJapaneseModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/modeling_gptsan_japanese.py#L851)

( config: GPTSanJapaneseConfig  )

Parameters

* **config** ([GPTSanJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.

The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
based Japanese language model

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/modeling_gptsan_japanese.py#L881)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.FloatTensor] = None spout: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None head\_mask: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = False inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None output\_router\_logits: typing.Optional[bool] = None num\_precontext: typing.Optional[torch.LongTensor] = None  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. GPTSAN-japanese is a model that generates sentence
  continuations or predicts tokens at mask positions. Special tokens required for inputs to the model are
  automatically appended.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  An input that masks the Prefix part in the Prefix-LM input. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **prefix** input,
  + 0 for tokens that are **not-prefix** input.
* **spout** (`torch.Tensor` of shape `(batch_size, config.d_spout)`) —
  This vector is transformed through an 8-layer FFN and can be used instead of `past_key_values`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **router\_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
  Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.
* **num\_precontext** (`torch.LongTensor` of shape `(batch_size,1)`) —
  length of `hybrid` input tokens in the input. Tokens up to this length refer to both front and back like
  BERT, tokens after that refer only to front like GPT. see also:
  <https://github.com/tanreinama/GPTSAN/blob/main/report/model.md>

The [GPTSanJapaneseModel](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## GPTSanJapaneseForConditionalGeneration

### class transformers.GPTSanJapaneseForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/modeling_gptsan_japanese.py#L1096)

( config: GPTSanJapaneseConfig  )

Parameters

* **config** ([GPTSanJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPTSAN-japanese Model with a language modeling head.

The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
based Japanese language model

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/gptsan_japanese/modeling_gptsan_japanese.py#L1107)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.FloatTensor] = None spout: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None head\_mask: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = False inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None output\_router\_logits: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. GPTSAN-japanese is a model that generates sentence
  continuations or predicts tokens at mask positions. Special tokens required for inputs to the model are
  automatically appended.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  An input that masks the Prefix part in the Prefix-LM input. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **prefix** input,
  + 0 for tokens that are **not-prefix** input.
* **spout** (`torch.Tensor` of shape `(batch_size, config.d_spout)`) —
  This vector is transformed through an 8-layer FFN and can be used instead of `past_key_values`.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **router\_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
  Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`

The [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

Text Generation with regular LM Model


```
>>> from transformers import AutoModel, AutoTokenizer, trainer_utils

>>> device = "cuda"
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> x_token = tokenizer("織田信長は、", return_tensors="pt")
>>> trainer_utils.set_seed(30)
>>> input_ids = x_token.input_ids.to(device)
>>> gen_token = model.generate(input_ids, max_new_tokens=50)
>>> tokenizer.decode(gen_token[0])
"織田信長は、政治・軍事の中枢まで掌握した政治家であり、日本史上類を見ない驚異的な軍事侵攻を続け..."
```

Text Generation with Prefix-LM Model


```
>>> from transformers import AutoModel, AutoTokenizer, trainer_utils

>>> device = "cuda"
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> x_token = tokenizer("", prefix_text="織田信長は、", return_tensors="pt")
>>> trainer_utils.set_seed(30)
>>> input_ids = x_token.input_ids.to(device)
>>> token_type_ids = x_token.token_type_ids.to(device)
>>> gen_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
>>> tokenizer.decode(gen_token[0])
"織田信長は、政治・外交で数々の戦果を上げるが、1568年からは、いわゆる本能寺の変で細川晴元に暗殺される..."
```

Simultaneously Text Generation And Masked Language Model


```
>>> from transformers import AutoModel, AutoTokenizer, trainer_utils

>>> device = "cuda"
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> masked_sentence = "武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。"
>>> x_token = tokenizer("", prefix_text=masked_sentence, return_tensors="pt")
>>> trainer_utils.set_seed(30)
>>> input_ids = x_token.input_ids.to(device)
>>> token_type_ids = x_token.token_type_ids.to(device)
>>> out_lm_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
>>> out_mlm_token = model(input_ids, token_type_ids=token_type_ids).logits.argmax(axis=-1)
>>> tokenizer.decode(out_mlm_token[0])
"武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。"

>>> tokenizer.decode(out_lm_token[0][input_ids.shape[1] :])
"武田氏の三代に渡った武田家のひとり\n甲斐市に住む、日本史上最大の戦国大名。..."
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gptsan-japanese.md)
