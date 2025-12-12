*This model was released on 2019-06-19 and added to Hugging Face Transformers on 2020-11-16.*

# XLNet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The XLNet model was proposed in [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://huggingface.co/papers/1906.08237) by Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov,
Quoc V. Le. XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn
bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization
order.

The abstract from the paper is the following:

*With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves
better performance than pretraining approaches based on autoregressive language modeling. However, relying on
corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a
pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive
pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all
permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive
formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into
pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large
margin, including question answering, natural language inference, sentiment analysis, and document ranking.*

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/zihangdai/xlnet/).

## Usage tips

* The specific attention pattern can be controlled at training and test time using the `perm_mask` input.
* Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained
  using only a sub-set of the output tokens as target which are selected with the `target_mapping` input.
* To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
  `target_mapping` inputs to control the attention span and outputs (see examples in
  *examples/pytorch/text-generation/run\_generation.py*)
* XLNet is one of the few models that has no sequence length limit.
* XLNet is not a traditional autoregressive model but uses a training strategy that builds on that. It permutes the tokens in the sentence, then allows the model to use the last n tokens to predict the token n+1. Since this is all done with a mask, the sentence is actually fed in the model in the right order, but instead of masking the first n tokens for n+1, XLNet uses a mask that hides the previous tokens in some given permutation of 1,…,sequence length.
* XLNet also uses the same recurrence mechanism as Transformer-XL to build long-term dependencies.

## Resources

* [Text classification task guide](../tasks/sequence_classification)
* [Token classification task guide](../tasks/token_classification)
* [Question answering task guide](../tasks/question_answering)
* [Causal language modeling task guide](../tasks/language_modeling)
* [Multiple choice task guide](../tasks/multiple_choice)

## XLNetConfig

### class transformers.XLNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/configuration_xlnet.py#L27)

( vocab\_size = 32000 d\_model = 1024 n\_layer = 24 n\_head = 16 d\_inner = 4096 ff\_activation = 'gelu' untie\_r = True attn\_type = 'bi' initializer\_range = 0.02 layer\_norm\_eps = 1e-12 dropout = 0.1 mem\_len = 512 reuse\_len = None use\_mems\_eval = True use\_mems\_train = False bi\_data = False clamp\_len = -1 same\_length = False summary\_type = 'last' summary\_use\_proj = True summary\_activation = 'tanh' summary\_last\_dropout = 0.1 start\_n\_top = 5 end\_n\_top = 5 pad\_token\_id = 5 bos\_token\_id = 1 eos\_token\_id = 2 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel) or `TFXLNetModel`.
* **d\_model** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers and the pooler layer.
* **n\_layer** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **n\_head** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **d\_inner** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **ff\_activation** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the If string, `"gelu"`, `"relu"`, `"silu"` and
  `"gelu_new"` are supported.
* **untie\_r** (`bool`, *optional*, defaults to `True`) —
  Whether or not to untie relative position biases
* **attn\_type** (`str`, *optional*, defaults to `"bi"`) —
  The attention type used by the model. Set `"bi"` for XLNet, `"uni"` for Transformer-XL.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **mem\_len** (`int` or `None`, *optional*) —
  The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
  forward pass won’t be re-computed. See the
  [quickstart](https://huggingface.co/transformers/quickstart.html#using-the-past) for more information.
* **reuse\_len** (`int`, *optional*) —
  The number of tokens in the current batch to be cached and reused in the future.
* **bi\_data** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use bidirectional input pipeline. Usually set to `True` during pretraining and `False`
  during finetuning.
* **clamp\_len** (`int`, *optional*, defaults to -1) —
  Clamp all relative distances larger than clamp\_len. Setting this attribute to -1 means no clamping.
* **same\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use the same attention length for each token.
* **summary\_type** (`str`, *optional*, defaults to “last”) —
  Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

  Has to be one of the following options:

  + `"last"`: Take the last token hidden state (like XLNet).
  + `"first"`: Take the first token hidden state (like BERT).
  + `"mean"`: Take the mean of all tokens hidden states.
  + `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
  + `"attn"`: Not implemented now, use multi-head attention.
* **summary\_use\_proj** (`bool`, *optional*, defaults to `True`) —
  Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

  Whether or not to add a projection after the vector extraction.
* **summary\_activation** (`str`, *optional*) —
  Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

  Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
* **summary\_proj\_to\_labels** (`boo`, *optional*, defaults to `True`) —
  Used in the sequence classification and multiple choice models.

  Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
* **summary\_last\_dropout** (`float`, *optional*, defaults to 0.1) —
  Used in the sequence classification and multiple choice models.

  The dropout ratio to be used after the projection and activation.
* **start\_n\_top** (`int`, *optional*, defaults to 5) —
  Used in the SQuAD evaluation script.
* **end\_n\_top** (`int`, *optional*, defaults to 5) —
  Used in the SQuAD evaluation script.
* **use\_mems\_eval** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.
* **use\_mems\_train** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should make use of the recurrent memory mechanism in train mode.

  For pretraining, it is recommended to set `use_mems_train` to `True`. For fine-tuning, it is recommended to
  set `use_mems_train` to `False` as discussed
  [here](https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587). If `use_mems_train` is set to
  `True`, one has to make sure that the train batches are correctly pre-processed, *e.g.* `batch_1 = [[This line is], [This is the]]` and `batch_2 = [[ the first line], [ second line]]` and that all batches are of
  equal size.

This is the configuration class to store the configuration of a [XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel) or a `TFXLNetModel`. It is used to
instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
[xlnet/xlnet-large-cased](https://huggingface.co/xlnet/xlnet-large-cased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import XLNetConfig, XLNetModel

>>> # Initializing a XLNet configuration
>>> configuration = XLNetConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = XLNetModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## XLNetTokenizer

### class transformers.XLNetTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L43)

( vocab\_file do\_lower\_case = False remove\_space = True keep\_accents = False bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' sep\_token = '<sep>' pad\_token = '<pad>' cls\_token = '<cls>' mask\_token = '<mask>' additional\_special\_tokens = ['<eop>', '<eod>'] sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **do\_lower\_case** (`bool`, *optional*, defaults to `False`) —
  Whether to lowercase the input when tokenizing.
* **remove\_space** (`bool`, *optional*, defaults to `True`) —
  Whether to strip the text when tokenizing (removing excess spaces before and after the string).
* **keep\_accents** (`bool`, *optional*, defaults to `False`) —
  Whether to keep accents when tokenizing.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"<sep>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"<cls>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `['<eop>', '<eod>']`) —
  Additional special tokens used by the tokenizer.
* **sp\_model\_kwargs** (`dict`, *optional*) —
  Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
  SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
  to set:
  + `enable_sampling`: Enable subword regularization.
  + `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

    - `nbest_size = {0,1}`: No sampling is performed.
    - `nbest_size > 1`: samples from the nbest\_size results.
    - `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice)
      using forward-filtering-and-backward-sampling algorithm.
  + `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
    BPE-dropout.
* **sp\_model** (`SentencePieceProcessor`) —
  The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

Construct an XLNet tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L286)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLNet sequence has the following format:

* single sequence: `X <sep> <cls>`
* pair of sequences: `A <sep> B <sep> <cls>`

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L311)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`list[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L339)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:


```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L369)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## XLNetTokenizerFast

### class transformers.XLNetTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L47)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = False remove\_space = True keep\_accents = False bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' sep\_token = '<sep>' pad\_token = '<pad>' cls\_token = '<cls>' mask\_token = '<mask>' additional\_special\_tokens = ['<eop>', '<eod>'] \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether to lowercase the input when tokenizing.
* **remove\_space** (`bool`, *optional*, defaults to `True`) —
  Whether to strip the text when tokenizing (removing excess spaces before and after the string).
* **keep\_accents** (`bool`, *optional*, defaults to `False`) —
  Whether to keep accents when tokenizing.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"<sep>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"<cls>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `["<eop>", "<eod>"]`) —
  Additional special tokens used by the tokenizer.
* **sp\_model** (`SentencePieceProcessor`) —
  The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

Construct a “fast” XLNet tokenizer (backed by HuggingFace’s *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L155)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLNet sequence has the following format:

* single sequence: `X <sep> <cls>`
* pair of sequences: `A <sep> B <sep> <cls>`

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L180)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet

sequence pair mask has the following format:


```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## XLNet specific outputs

### class transformers.models.xlnet.modeling\_xlnet.XLNetModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L841)

( last\_hidden\_state: FloatTensor mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) —
  Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel).

### class transformers.models.xlnet.modeling\_xlnet.XLNetLMHeadModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L866)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) —
  Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`) —
  Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForSequenceClassificationOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L894)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) —
  Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) —
  Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [XLNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForMultipleChoiceOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L944)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) —
  Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) —
  *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [XLNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForTokenClassificationOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L919)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) —
  Classification scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of `XLNetForTokenClassificationOutput`.

### class transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringSimpleOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L971)

( loss: typing.Optional[torch.FloatTensor] = None start\_logits: typing.Optional[torch.FloatTensor] = None end\_logits: typing.Optional[torch.FloatTensor] = None mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) —
  Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) —
  Span-end scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [XLNetForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple).

### class transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L999)

( loss: typing.Optional[torch.FloatTensor] = None start\_top\_log\_probs: typing.Optional[torch.FloatTensor] = None start\_top\_index: typing.Optional[torch.LongTensor] = None end\_top\_log\_probs: typing.Optional[torch.FloatTensor] = None end\_top\_index: typing.Optional[torch.LongTensor] = None cls\_logits: typing.Optional[torch.FloatTensor] = None mems: typing.Optional[list[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) —
  Classification loss as the sum of start token, end token (and is\_impossible if provided) classification
  losses.
* **start\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Log probabilities for the top config.start\_n\_top start token possibilities (beam-search).
* **start\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Indices for the top config.start\_n\_top start token possibilities (beam-search).
* **end\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
* **end\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
* **cls\_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Log probabilities for the `is_impossible` label of the answers.
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [XLNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering).

## XLNetModel

### class transformers.XLNetModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1033)

( config  )

Parameters

* **config** ([XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Xlnet Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1161)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetModelOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetModelOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetModelOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) — Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## XLNetLMHeadModel

### class transformers.XLNetLMHeadModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1424)

( config  )

Parameters

* **config** ([XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1488)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetLMHeadModelOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, num_predict)`, *optional*) —
  Labels for masked language modeling. `num_predict` corresponds to `target_mapping.shape[1]`. If
  `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

  The labels should correspond to the masked input words that should be predicted and depends on
  `target_mapping`. Note in order to perform standard auto-regressive language modeling a  token has
  to be added to the `input_ids` (see the `prepare_inputs_for_generation` function and examples below)

  Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored, the loss
  is only computed for labels in `[0, ..., config.vocab_size]`
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetLMHeadModelOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetLMHeadModelOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, XLNetLMHeadModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-large-cased")

>>> # We show how to setup inputs to predict a next token using a bi-directional context.
>>> input_ids = torch.tensor(
...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
... ).unsqueeze(
...     0
... )  # We will predict the masked token
>>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
>>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
>>> target_mapping = torch.zeros(
...     (1, 1, input_ids.shape[1]), dtype=torch.float
... )  # Shape [1, 1, seq_length] => let's predict one token
>>> target_mapping[
...     0, 0, -1
... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

>>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
>>> next_token_logits = outputs[
...     0
... ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

>>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
>>> input_ids = torch.tensor(
...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
... ).unsqueeze(
...     0
... )  # We will predict the masked token
>>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
>>> assert labels.shape[0] == 1, "only one word will be predicted"
>>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
>>> perm_mask[
...     :, :, -1
... ] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
>>> target_mapping = torch.zeros(
...     (1, 1, input_ids.shape[1]), dtype=torch.float
... )  # Shape [1, 1, seq_length] => let's predict one token
>>> target_mapping[
...     0, 0, -1
... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

>>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
>>> loss = outputs.loss
>>> next_token_logits = (
...     outputs.logits
... )  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
```

## XLNetForSequenceClassification

### class transformers.XLNetForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1660)

( config  )

Parameters

* **config** ([XLNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1673)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, XLNetForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetForSequenceClassification.from_pretrained("xlnet/xlnet-large-cased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = XLNetForSequenceClassification.from_pretrained("xlnet/xlnet-large-cased", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, XLNetForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetForSequenceClassification.from_pretrained("xlnet/xlnet-large-cased", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = XLNetForSequenceClassification.from_pretrained(
...     "xlnet/xlnet-large-cased", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## XLNetForMultipleChoice

### class transformers.XLNetForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1900)

( config  )

Parameters

* **config** ([XLNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlnet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1911)

( input\_ids: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, num_choices, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, …,
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLNetForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetForMultipleChoice.from_pretrained("xlnet/xlnet-large-cased")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

>>> # the linear classifier still needs to be trained
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## XLNetForTokenClassification

### class transformers.XLNetForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1790)

( config  )

Parameters

* **config** ([XLNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlnet transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1801)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
  where *num\_choices* is the size of the second dimension of the input tensors. (see *input\_ids* above)
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.emory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLNetForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetForTokenClassification.from_pretrained("xlnet/xlnet-large-cased")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
...

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

## XLNetForQuestionAnsweringSimple

### class transformers.XLNetForQuestionAnsweringSimple

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2046)

( config  )

Parameters

* **config** ([XLNetForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2057)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) — Span-end scores (before SoftMax).
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLNetForQuestionAnsweringSimple
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet/xlnet-large-cased")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
...

>>> # target is "nice puppet"
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
...
```

## XLNetForQuestionAnswering

### class transformers.XLNetForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2169)

( config  )

Parameters

* **config** ([XLNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlnet transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2183)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None mems: typing.Optional[torch.Tensor] = None perm\_mask: typing.Optional[torch.Tensor] = None target\_mapping: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None input\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None is\_impossible: typing.Optional[torch.Tensor] = None cls\_index: typing.Optional[torch.Tensor] = None p\_mask: typing.Optional[torch.Tensor] = None use\_mems: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) —
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
* **perm\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) —
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:
  + if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  + if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
* **target\_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) —
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **input\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) —
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  + 1 for tokens that are **masked**,
  + 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **is\_impossible** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels whether a question has an answer or no answer (SQuAD 2.0)
* **cls\_index** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the classification token to use as input for computing plausibility of the
  answer.
* **p\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Optional mask of tokens which can’t be in answers (e.g. [CLS], [PAD], …). 1.0 means token should be
  masked. 0.0 mean token is not masked.
* **use\_mems** (`bool`, *optional*) —
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlnet.modeling\_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) — Classification loss as the sum of start token, end token (and is\_impossible if provided) classification
  losses.
* **start\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Log probabilities for the top config.start\_n\_top start token possibilities (beam-search).
* **start\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Indices for the top config.start\_n\_top start token possibilities (beam-search).
* **end\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
* **end\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
* **cls\_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Log probabilities for the `is_impossible` label of the answers.
* **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLNetForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
>>> model = XLNetForQuestionAnswering.from_pretrained("xlnet/xlnet-base-cased")

>>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
...     0
... )  # Batch size 1
>>> start_positions = torch.tensor([1])
>>> end_positions = torch.tensor([3])
>>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

>>> loss = outputs.loss
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlnet.md)
