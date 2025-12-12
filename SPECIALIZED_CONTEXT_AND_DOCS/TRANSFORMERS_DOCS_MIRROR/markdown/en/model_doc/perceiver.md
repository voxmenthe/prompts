*This model was released on 2021-07-30 and added to Hugging Face Transformers on 2021-12-08.*

# Perceiver

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Perceiver IO model was proposed in [Perceiver IO: A General Architecture for Structured Inputs &
Outputs](https://huggingface.co/papers/2107.14795) by Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch,
Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M.
Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira.

Perceiver IO is a generalization of [Perceiver](https://huggingface.co/papers/2103.03206) to handle arbitrary outputs in
addition to arbitrary inputs. The original Perceiver only produced a single classification label. In addition to
classification labels, Perceiver IO can produce (for example) language, optical flow, and multimodal videos with audio.
This is done using the same building blocks as the original Perceiver. The computational complexity of Perceiver IO is
linear in the input and output size and the bulk of the processing occurs in the latent space, allowing us to process
inputs and outputs that are much larger than can be handled by standard Transformers. This means, for example,
Perceiver IO can do BERT-style masked language modeling directly using bytes instead of tokenized inputs.

The abstract from the paper is the following:

*The recently-proposed Perceiver model obtains good results on several domains (images, audio, multimodal, point
clouds) while scaling linearly in compute and memory with the input size. While the Perceiver supports many kinds of
inputs, it can only produce very simple outputs such as class scores. Perceiver IO overcomes this limitation without
sacrificing the original’s appealing properties by learning to flexibly query the model’s latent space to produce
outputs of arbitrary size and semantics. Perceiver IO still decouples model depth from data size and still scales
linearly with data size, but now with respect to both input and output sizes. The full Perceiver IO model achieves
strong results on tasks with highly structured output spaces, such as natural language and visual understanding,
StarCraft II, and multi-task and multi-modal domains. As highlights, Perceiver IO matches a Transformer-based BERT
baseline on the GLUE language benchmark without the need for input tokenization and achieves state-of-the-art
performance on Sintel optical flow estimation.*

Here’s a TLDR explaining how Perceiver works:

The main problem with the self-attention mechanism of the Transformer is that the time and memory requirements scale
quadratically with the sequence length. Hence, models like BERT and RoBERTa are limited to a max sequence length of 512
tokens. Perceiver aims to solve this issue by, instead of performing self-attention on the inputs, perform it on a set
of latent variables, and only use the inputs for cross-attention. In this way, the time and memory requirements don’t
depend on the length of the inputs anymore, as one uses a fixed amount of latent variables, like 256 or 512. These are
randomly initialized, after which they are trained end-to-end using backpropagation.

Internally, [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) will create the latents, which is a tensor of shape `(batch_size, num_latents, d_latents)`. One must provide `inputs` (which could be text, images, audio, you name it!) to the model, which it will
use to perform cross-attention with the latents. The output of the Perceiver encoder is a tensor of the same shape. One
can then, similar to BERT, convert the last hidden states of the latents to classification logits by averaging along
the sequence dimension, and placing a linear layer on top of that to project the `d_latents` to `num_labels`.

This was the idea of the original Perceiver paper. However, it could only output classification logits. In a follow-up
work, PerceiverIO, they generalized it to let the model also produce outputs of arbitrary size. How, you might ask? The
idea is actually relatively simple: one defines outputs of an arbitrary size, and then applies cross-attention with the
last hidden states of the latents, using the outputs as queries, and the latents as keys and values.

So let’s say one wants to perform masked language modeling (BERT-style) with the Perceiver. As the Perceiver’s input
length will not have an impact on the computation time of the self-attention layers, one can provide raw bytes,
providing `inputs` of length 2048 to the model. If one now masks out certain of these 2048 tokens, one can define the
`outputs` as being of shape: `(batch_size, 2048, 768)`. Next, one performs cross-attention with the final hidden states
of the latents to update the `outputs` tensor. After cross-attention, one still has a tensor of shape `(batch_size, 2048, 768)`. One can then place a regular language modeling head on top, to project the last dimension to the
vocabulary size of the model, i.e. creating logits of shape `(batch_size, 2048, 262)` (as Perceiver uses a vocabulary
size of 262 byte IDs).

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perceiver_architecture.jpg) Perceiver IO architecture. Taken from the [original paper](https://huggingface.co/papers/2105.15203)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/deepmind/deepmind-research/tree/master/perceiver).

Perceiver does **not** work with `torch.nn.DataParallel` due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035)

## Resources

* The quickest way to get started with the Perceiver is by checking the [tutorial
  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Perceiver).
* Refer to the [blog post](https://huggingface.co/blog/perceiver) if you want to fully understand how the model works and
  is implemented in the library. Note that the models available in the library only showcase some examples of what you can do
  with the Perceiver. There are many more use cases, including question answering, named-entity recognition, object detection,
  audio classification, video classification, etc.
* [Text classification task guide](../tasks/sequence_classification)
* [Masked language modeling task guide](../tasks/masked_language_modeling)
* [Image classification task guide](../tasks/image_classification)

## Perceiver specific outputs

### class transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L53)

( logits: typing.Optional[torch.FloatTensor] = None last\_hidden\_state: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **logits** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) —
  Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) —
  Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for Perceiver base model’s outputs, with potential hidden states, attentions and cross-attentions.

### class transformers.models.perceiver.modeling\_perceiver.PerceiverDecoderOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L72)

( logits: typing.Optional[torch.FloatTensor] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **logits** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) —
  Output of the basic decoder.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for Perceiver decoder outputs, with potential cross-attentions.

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L88)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) —
  Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for Perceiver’s masked language model outputs.

### class transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L110)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) —
  Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for Perceiver’s outputs of sequence/image classification models, optical flow and multimodal
autoencoding.

## PerceiverConfig

### class transformers.PerceiverConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/configuration_perceiver.py#L32)

( num\_latents = 256 d\_latents = 1280 d\_model = 768 num\_blocks = 1 num\_self\_attends\_per\_block = 26 num\_self\_attention\_heads = 8 num\_cross\_attention\_heads = 8 qk\_channels = None v\_channels = None cross\_attention\_shape\_for\_attention = 'kv' self\_attention\_widening\_factor = 1 cross\_attention\_widening\_factor = 1 hidden\_act = 'gelu' attention\_probs\_dropout\_prob = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 use\_query\_residual = True vocab\_size = 262 max\_position\_embeddings = 2048 image\_size = 56 train\_size = [368, 496] num\_frames = 16 audio\_samples\_per\_frame = 1920 samples\_per\_patch = 16 output\_shape = [1, 16, 224, 224] output\_num\_channels = 512 \_label\_trainable\_num\_channels = 1024 \*\*kwargs  )

Parameters

* **num\_latents** (`int`, *optional*, defaults to 256) —
  The number of latents.
* **d\_latents** (`int`, *optional*, defaults to 1280) —
  Dimension of the latent embeddings.
* **d\_model** (`int`, *optional*, defaults to 768) —
  Dimension of the inputs. Should only be provided in case [*PerceiverTextPreprocessor*] is used or no
  preprocessor is provided.
* **num\_blocks** (`int`, *optional*, defaults to 1) —
  Number of blocks in the Transformer encoder.
* **num\_self\_attends\_per\_block** (`int`, *optional*, defaults to 26) —
  The number of self-attention layers per block.
* **num\_self\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each self-attention layer in the Transformer encoder.
* **num\_cross\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each cross-attention layer in the Transformer encoder.
* **qk\_channels** (`int`, *optional*) —
  Dimension to project the queries + keys before applying attention in the cross-attention and self-attention
  layers of the encoder. Will default to preserving the dimension of the queries if not specified.
* **v\_channels** (`int`, *optional*) —
  Dimension to project the values before applying attention in the cross-attention and self-attention layers
  of the encoder. Will default to preserving the dimension of the queries if not specified.
* **cross\_attention\_shape\_for\_attention** (`str`, *optional*, defaults to `"kv"`) —
  Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
* **self\_attention\_widening\_factor** (`int`, *optional*, defaults to 1) —
  Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
* **cross\_attention\_widening\_factor** (`int`, *optional*, defaults to 1) —
  Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **use\_query\_residual** (`float`, *optional*, defaults to `True`) —
  Whether to add a query residual in the cross-attention layer of the encoder.
* **vocab\_size** (`int`, *optional*, defaults to 262) —
  Vocabulary size for the masked language modeling model.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 2048) —
  The maximum sequence length that the masked language modeling model might ever be used with. Typically set
  this to something large just in case (e.g., 512 or 1024 or 2048).
* **image\_size** (`int`, *optional*, defaults to 56) —
  Size of the images after preprocessing, for [PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned).
* **train\_size** (`list[int]`, *optional*, defaults to `[368, 496]`) —
  Training size of the images for the optical flow model.
* **num\_frames** (`int`, *optional*, defaults to 16) —
  Number of video frames used for the multimodal autoencoding model.
* **audio\_samples\_per\_frame** (`int`, *optional*, defaults to 1920) —
  Number of audio samples per frame for the multimodal autoencoding model.
* **samples\_per\_patch** (`int`, *optional*, defaults to 16) —
  Number of audio samples per patch when preprocessing the audio for the multimodal autoencoding model.
* **output\_shape** (`list[int]`, *optional*, defaults to `[1, 16, 224, 224]`) —
  Shape of the output (batch\_size, num\_frames, height, width) for the video decoder queries of the multimodal
  autoencoding model. This excludes the channel dimension.
* **output\_num\_channels** (`int`, *optional*, defaults to 512) —
  Number of output channels for each modalitiy decoder.

This is the configuration class to store the configuration of a [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel). It is used to instantiate an
Perceiver model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Perceiver
[deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PerceiverModel, PerceiverConfig

>>> # Initializing a Perceiver deepmind/language-perceiver style configuration
>>> configuration = PerceiverConfig()

>>> # Initializing a model from the deepmind/language-perceiver style configuration
>>> model = PerceiverModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PerceiverTokenizer

### class transformers.PerceiverTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/tokenization_perceiver.py#L26)

( pad\_token = '[PAD]' bos\_token = '[BOS]' eos\_token = '[EOS]' mask\_token = '[MASK]' cls\_token = '[CLS]' sep\_token = '[SEP]' model\_max\_length = 2048 \*\*kwargs  )

Parameters

* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **bos\_token** (`str`, *optional*, defaults to `"[BOS]"`) —
  The BOS token (reserved in the vocab, but not actually used).
* **eos\_token** (`str`, *optional*, defaults to `"[EOS]"`) —
  The end of sequence token (reserved in the vocab, but not actually used).

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The MASK token, useful for masked language modeling.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The CLS token (reserved in the vocab, but not actually used).
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from two sequences.

Construct a Perceiver tokenizer. The Perceiver simply uses raw bytes utf-8 encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828)

( text: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
* **return\_special\_tokens\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using
  Python’s tokenizer, this method will raise `NotImplementedError`.
* **return\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the lengths of the encoded inputs.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.
* \***\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

## PerceiverFeatureExtractor

### class transformers.PerceiverFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/feature_extraction_perceiver.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## PerceiverImageProcessor

### class transformers.PerceiverImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/image_processing_perceiver.py#L49)

( do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_center\_crop** (`bool`, `optional`, defaults to `True`) —
  Whether or not to center crop the image. If the input size if smaller than `crop_size` along any edge, the
  image will be padded with zeros and then center cropped. Can be overridden by the `do_center_crop`
  parameter in the `preprocess` method.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 256, "width": 256}`):
  Desired output size when applying center-cropping. Can be overridden by the `crop_size` parameter in the
  `preprocess` method.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image to `(size["height"], size["width"])`. Can be overridden by the `do_resize`
  parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of the image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter
  in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
  in the `preprocess` method.
* **do\_normalize** —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a Perceiver image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/image_processing_perceiver.py#L212)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the image to `crop_size`.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Desired output size after applying the center crop.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## PerceiverImageProcessorFast

### class transformers.PerceiverImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/image_processing_perceiver_fast.py#L42)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Perceiver image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L639)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*args \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## PerceiverTextPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverTextPreprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2776)

( config: PerceiverConfig  )

Parameters

* **config** ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) —
  Model configuration.

Text preprocessing for Perceiver Encoder. Can be used to embed `inputs` and add positional encodings.

The dimensionality of the embeddings is determined by the `d_model` attribute of the configuration.

## PerceiverImagePreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverImagePreprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2939)

( config prep\_type = 'conv' spatial\_downsample: int = 4 temporal\_downsample: int = 1 position\_encoding\_type: str = 'fourier' in\_channels: int = 3 out\_channels: int = 64 conv\_after\_patching: bool = False conv\_after\_patching\_in\_channels: int = 54 conv2d\_use\_batchnorm: bool = True concat\_or\_add\_pos: str = 'concat' project\_pos\_dim: int = -1 \*\*position\_encoding\_kwargs  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **prep\_type** (`str`, *optional*, defaults to `"conv"`) —
  Preprocessing type. Can be “conv1x1”, “conv”, “patches”, “pixels”.
* **spatial\_downsample** (`int`, *optional*, defaults to 4) —
  Spatial downsampling factor.
* **temporal\_downsample** (`int`, *optional*, defaults to 1) —
  Temporal downsampling factor (only relevant in case a time dimension is present).
* **position\_encoding\_type** (`str`, *optional*, defaults to `"fourier"`) —
  Position encoding type. Can be “fourier” or “trainable”.
* **in\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input.
* **out\_channels** (`int`, *optional*, defaults to 64) —
  Number of channels in the output.
* **conv\_after\_patching** (`bool`, *optional*, defaults to `False`) —
  Whether to apply a convolutional layer after patching.
* **conv\_after\_patching\_in\_channels** (`int`, *optional*, defaults to 54) —
  Number of channels in the input of the convolutional layer after patching.
* **conv2d\_use\_batchnorm** (`bool`, *optional*, defaults to `True`) —
  Whether to use batch normalization in the convolutional layer.
* **concat\_or\_add\_pos** (`str`, *optional*, defaults to `"concat"`) —
  How to concatenate the position encoding to the input. Can be “concat” or “add”.
* **project\_pos\_dim** (`int`, *optional*, defaults to -1) —
  Dimension of the position encoding to project to. If -1, no projection is applied.
* \***\*position\_encoding\_kwargs** (`Dict`, *optional*) —
  Keyword arguments for the position encoding.

Image preprocessing for Perceiver Encoder.

Note: the *out\_channels* argument refers to the output channels of a convolutional layer, if *prep\_type* is set to
“conv1x1” or “conv”. If one adds absolute position embeddings, one must make sure the *num\_channels* of the
position encoding kwargs are set equal to the *out\_channels*.

## PerceiverOneHotPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverOneHotPreprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L3177)

( config: PerceiverConfig  )

Parameters

* **config** ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) —
  Model configuration.

One-hot preprocessor for Perceiver Encoder. Can be used to add a dummy index dimension to the input.

## PerceiverAudioPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverAudioPreprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L3203)

( config prep\_type: str = 'patches' samples\_per\_patch: int = 96 position\_encoding\_type: str = 'fourier' concat\_or\_add\_pos: str = 'concat' out\_channels = 64 project\_pos\_dim = -1 \*\*position\_encoding\_kwargs  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **prep\_type** (`str`, *optional*, defaults to `"patches"`) —
  Preprocessor type to use. Only “patches” is supported.
* **samples\_per\_patch** (`int`, *optional*, defaults to 96) —
  Number of samples per patch.
* **position\_encoding\_type** (`str`, *optional*, defaults to `"fourier"`) —
  Type of position encoding to use. Can be “trainable” or “fourier”.
* **concat\_or\_add\_pos** (`str`, *optional*, defaults to `"concat"`) —
  How to concatenate the position encoding to the input. Can be “concat” or “add”.
* **out\_channels** (`int`, *optional*, defaults to 64) —
  Number of channels in the output.
* **project\_pos\_dim** (`int`, *optional*, defaults to -1) —
  Dimension of the position encoding to project to. If -1, no projection is applied.
* \***\*position\_encoding\_kwargs** (`Dict`, *optional*) —
  Keyword arguments for the position encoding.

Audio preprocessing for Perceiver Encoder.

## PerceiverMultimodalPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalPreprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L3306)

( modalities: Mapping mask\_probs: typing.Optional[collections.abc.Mapping[str, float]] = None min\_padding\_size: int = 2  )

Parameters

* **modalities** (`Mapping[str, PreprocessorType]`) —
  Dict mapping modality name to preprocessor.
* **mask\_probs** (`dict[str, float]`) —
  Dict mapping modality name to masking probability of that modality.
* **min\_padding\_size** (`int`, *optional*, defaults to 2) —
  The minimum padding size for all modalities. The final output will have num\_channels equal to the maximum
  channels across all modalities plus min\_padding\_size.

Multimodal preprocessing for Perceiver Encoder.

Inputs for each modality are preprocessed, then padded with trainable position embeddings to have the same number
of channels.

## PerceiverProjectionDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverProjectionDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1954)

( config  )

Parameters

* **config** ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) —
  Model configuration.

Baseline projection decoder (no cross-attention).

## PerceiverBasicDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverBasicDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1980)

( config: PerceiverConfig output\_num\_channels: int position\_encoding\_type: typing.Optional[str] = 'trainable' output\_index\_dims: typing.Optional[int] = None num\_channels: typing.Optional[int] = 128 subsampled\_index\_dims: typing.Optional[int] = None qk\_channels: typing.Optional[int] = None v\_channels: typing.Optional[int] = None num\_heads: typing.Optional[int] = 1 widening\_factor: typing.Optional[int] = 1 use\_query\_residual: typing.Optional[bool] = False concat\_preprocessed\_input: typing.Optional[bool] = False final\_project: typing.Optional[bool] = True position\_encoding\_only: typing.Optional[bool] = False \*\*position\_encoding\_kwargs  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **output\_num\_channels** (`int`, *optional*) —
  The number of channels in the output. Will only be used in case *final\_project* is set to `True`.
* **position\_encoding\_type** (`str`, *optional*, defaults to “trainable”) —
  The type of position encoding to use. Can be either “trainable”, “fourier”, or “none”.
* **output\_index\_dims** (`int`, *optional*) —
  The number of dimensions of the output queries. Ignored if ‘position\_encoding\_type’ == ‘none’.
* **num\_channels** (`int`, *optional*, defaults to 128) —
  The number of channels of the decoder queries. Ignored if ‘position\_encoding\_type’ == ‘none’.
* **qk\_channels** (`int`, *optional*) —
  The number of channels of the queries and keys in the cross-attention layer.
* **v\_channels** (`int`, *optional*) —
  The number of channels of the values in the cross-attention layer.
* **num\_heads** (`int`, *optional*, defaults to 1) —
  The number of attention heads in the cross-attention layer.
* **widening\_factor** (`int`, *optional*, defaults to 1) —
  The widening factor of the cross-attention layer.
* **use\_query\_residual** (`bool`, *optional*, defaults to `False`) —
  Whether to use a residual connection between the query and the output of the cross-attention layer.
* **concat\_preprocessed\_input** (`bool`, *optional*, defaults to `False`) —
  Whether to concatenate the preprocessed input to the query.
* **final\_project** (`bool`, *optional*, defaults to `True`) —
  Whether to project the output of the cross-attention layer to a target dimension.
* **position\_encoding\_only** (`bool`, *optional*, defaults to `False`) —
  Whether to only use this class to define output queries.

Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
cross-attention operation, in which the latents produce keys and values.

The shape of the output of this class depends on how one defines the output queries (also called decoder queries).

## PerceiverClassificationDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverClassificationDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2166)

( config \*\*decoder\_kwargs  )

Parameters

* **config** ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) —
  Model configuration.

Cross-attention based classification decoder. Light-weight wrapper of `PerceiverBasicDecoder` for logit output.
Will turn the output of the Perceiver encoder which is of shape (batch\_size, num\_latents, d\_latents) to a tensor of
shape (batch\_size, num\_labels). The queries are of shape (batch\_size, 1, num\_labels).

## PerceiverOpticalFlowDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverOpticalFlowDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2212)

( config output\_image\_shape output\_num\_channels = 2 rescale\_factor = 100.0 \*\*decoder\_kwargs  )

Cross-attention based optical flow decoder.

## PerceiverBasicVideoAutoencodingDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverBasicVideoAutoencodingDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2247)

( config: PerceiverConfig output\_shape: list position\_encoding\_type: str \*\*decoder\_kwargs  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **output\_shape** (`list[int]`) —
  Shape of the output as (batch\_size, num\_frames, height, width), excluding the channel dimension.
* **position\_encoding\_type** (`str`) —
  The type of position encoding to use. Can be either “trainable”, “fourier”, or “none”.

Cross-attention based video-autoencoding decoder. Light-weight wrapper of [*PerceiverBasicDecoder*] with video
reshaping logic.

## PerceiverMultimodalDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2324)

( config: PerceiverConfig modalities: dict num\_outputs: int output\_num\_channels: int min\_padding\_size: typing.Optional[int] = 2 subsampled\_index\_dims: typing.Optional[dict[str, transformers.models.perceiver.modeling\_perceiver.PerceiverAbstractDecoder]] = None \*\*decoder\_kwargs  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **modalities** (`dict[str, PerceiverAbstractDecoder]`) —
  Dictionary mapping modality name to the decoder of that modality.
* **num\_outputs** (`int`) —
  The number of outputs of the decoder.
* **output\_num\_channels** (`int`) —
  The number of channels in the output.
* **min\_padding\_size** (`int`, *optional*, defaults to 2) —
  The minimum padding size for all modalities. The final output will have num\_channels equal to the maximum
  channels across all modalities plus min\_padding\_size.
* **subsampled\_index\_dims** (`dict[str, PerceiverAbstractDecoder]`, *optional*) —
  Dictionary mapping modality name to the subsampled index dimensions to use for the decoder query of that
  modality.

Multimodal decoding by composing uni-modal decoders. The *modalities* argument of the constructor is a dictionary
mapping modality name to the decoder of that modality. That decoder will be used to construct queries for that
modality. Modality-specific queries are padded with trainable modality-specific parameters, after which they are
concatenated along the time dimension.

Next, there is a shared cross attention operation across all modalities.

## PerceiverProjectionPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverProjectionPostprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2918)

( in\_channels: int out\_channels: int  )

Parameters

* **in\_channels** (`int`) —
  Number of channels in the input.
* **out\_channels** (`int`) —
  Number of channels in the output.

Projection postprocessing for Perceiver. Can be used to project the channels of the decoder output to a lower
dimension.

## PerceiverAudioPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverAudioPostprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2891)

( config: PerceiverConfig in\_channels: int postproc\_type: str = 'patches'  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **in\_channels** (`int`) —
  Number of channels in the input.
* **postproc\_type** (`str`, *optional*, defaults to `"patches"`) —
  Postprocessor type to use. Currently, only “patches” is supported.

Audio postprocessing for Perceiver. Can be used to convert the decoder output to audio features.

## PerceiverClassificationPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverClassificationPostprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2871)

( config: PerceiverConfig in\_channels: int  )

Parameters

* **config** ([*PerceiverConfig*]) —
  Model configuration.
* **in\_channels** (`int`) —
  Number of channels in the input.

Classification postprocessing for Perceiver. Can be used to convert the decoder output to classification logits.

## PerceiverMultimodalPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalPostprocessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L2837)

( modalities: Mapping input\_is\_dict: bool = False  )

Parameters

* **modalities** (`Mapping[str, PostprocessorType]`) —
  Dictionary mapping modality name to postprocessor class for that modality.
* **input\_is\_dict** (`bool`, *optional*, defaults to `False`) —
  If True, input is assumed to be dictionary structured, and outputs keep the same dictionary shape. If
  False, input is a tensor which is sliced up during postprocessing by *modality\_sizes*.

Multimodal postprocessing for Perceiver. Can be used to combine modality-specific postprocessors into a single
postprocessor.

## PerceiverModel

### class transformers.PerceiverModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L604)

( config decoder: typing.Optional[ForwardRef('PerceiverAbstractDecoder')] = None input\_preprocessor: typing.Callable[..., tuple[torch.Tensor, typing.Optional[torch.Tensor], torch.Tensor]] = None output\_postprocessor: typing.Callable[..., typing.Any] = None  )

Parameters

* **config** ([PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **decoder** (`PerceiverDecoder`, *optional*) —
  Decoder module that transforms latent representations into task predictions.
* **input\_preprocessor** (`PreprocessorType`, *optional*) —
  Preprocessor that encodes raw inputs into tensors for the model.
* **output\_postprocessor** (`PostprocessorType`, *optional*) —
  Postprocessor that transforms model outputs into final predictions.

The Perceiver: a scalable, fully attentional architecture.

Note that it’s possible to fine-tune Perceiver on higher resolution images than the ones it has been trained on, by
setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
position embeddings to the higher resolution.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L648)

( inputs: FloatTensor attention\_mask: typing.Optional[torch.FloatTensor] = None subsampled\_output\_points: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.FloatTensor`) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **subsampled\_output\_points** (`dict[str, torch.Tensor]`, *optional*) —
  Dictionary of tensors used as queries for the decoder. The decoder maps these queries to the latent
  representation of the model. Used for subsampled decoding, e.g. when only decoding certain image patches.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **logits** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
>>> from transformers.models.perceiver.modeling_perceiver import (
...     PerceiverTextPreprocessor,
...     PerceiverImagePreprocessor,
...     PerceiverClassificationDecoder,
... )
>>> import torch
>>> import requests
>>> from PIL import Image

>>> # EXAMPLE 1: using the Perceiver to classify texts
>>> # - we define a TextPreprocessor, which can be used to embed tokens
>>> # - we define a ClassificationDecoder, which can be used to decode the
>>> # final hidden states of the latents to classification logits
>>> # using trainable position embeddings
>>> config = PerceiverConfig()
>>> preprocessor = PerceiverTextPreprocessor(config)
>>> decoder = PerceiverClassificationDecoder(
...     config,
...     num_channels=config.d_latents,
...     trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
...     use_query_residual=True,
... )
>>> model = PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)

>>> # you can then do a forward pass as follows:
>>> tokenizer = PerceiverTokenizer()
>>> text = "hello world"
>>> inputs = tokenizer(text, return_tensors="pt").input_ids

>>> with torch.no_grad():
...     outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2]

>>> # to train, one can train the model using standard cross-entropy:
>>> criterion = torch.nn.CrossEntropyLoss()

>>> labels = torch.tensor([1])
>>> loss = criterion(logits, labels)

>>> # EXAMPLE 2: using the Perceiver to classify images
>>> # - we define an ImagePreprocessor, which can be used to embed images
>>> config = PerceiverConfig(image_size=224)
>>> preprocessor = PerceiverImagePreprocessor(
...     config,
...     prep_type="conv1x1",
...     spatial_downsample=1,
...     out_channels=256,
...     position_encoding_type="trainable",
...     concat_or_add_pos="concat",
...     project_pos_dim=256,
...     trainable_position_encoding_kwargs=dict(
...         num_channels=256,
...         index_dims=config.image_size**2,
...     ),
... )

>>> model = PerceiverModel(
...     config,
...     input_preprocessor=preprocessor,
...     decoder=PerceiverClassificationDecoder(
...         config,
...         num_channels=config.d_latents,
...         trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
...         use_query_residual=True,
...     ),
... )

>>> # you can then do a forward pass as follows:
>>> image_processor = PerceiverImageProcessor()
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(image, return_tensors="pt").pixel_values

>>> with torch.no_grad():
...     outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2]

>>> # to train, one can train the model using standard cross-entropy:
>>> criterion = torch.nn.CrossEntropyLoss()

>>> labels = torch.tensor([1])
>>> loss = criterion(logits, labels)
```

## PerceiverForMaskedLM

### class transformers.PerceiverForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L860)

( config: PerceiverConfig  )

Parameters

* **config** ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for masked language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L892)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None input\_ids: typing.Optional[torch.Tensor] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, PerceiverForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("deepmind/language-perceiver")
>>> model = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")

>>> # training
>>> text = "This is an incomplete sentence where some words are missing."
>>> inputs = tokenizer(text, padding="max_length", return_tensors="pt")
>>> # mask " missing."
>>> inputs["input_ids"][0, 52:61] = tokenizer.mask_token_id
>>> labels = tokenizer(text, padding="max_length", return_tensors="pt").input_ids

>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
19.87

>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2048, 262]

>>> # inference
>>> text = "This is an incomplete sentence where some words are missing."
>>> encoding = tokenizer(text, padding="max_length", return_tensors="pt")

>>> # mask bytes corresponding to " missing.". Note that the model performs much better if the masked span starts with a space.
>>> encoding["input_ids"][0, 52:61] = tokenizer.mask_token_id

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**encoding)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2048, 262]

>>> masked_tokens_predictions = logits[0, 52:61].argmax(dim=-1).tolist()
>>> tokenizer.decode(masked_tokens_predictions)
' missing.'
```

## PerceiverForSequenceClassification

### class transformers.PerceiverForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L998)

( config  )

Parameters

* **config** ([PerceiverForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for text classification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1019)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None input\_ids: typing.Optional[torch.Tensor] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, PerceiverForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("deepmind/language-perceiver")
>>> model = PerceiverForSequenceClassification.from_pretrained("deepmind/language-perceiver")

>>> text = "hello world"
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2]
```

## PerceiverForImageClassificationLearned

### class transformers.PerceiverForImageClassificationLearned

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1121)

( config  )

Parameters

* **config** ([PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses learned position embeddings. In other words, this model is not given any privileged information about
the structure of images. As shown in the paper, this model can achieve a top-1 accuracy of 72.7 on ImageNet.

[PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) uses [PerceiverImagePreprocessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor)
(with `prep_type="conv1x1"`) to preprocess the input images, and
[PerceiverClassificationDecoder](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder) to decode the latent representation of
[PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) into classification logits.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1152)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor). See [PerceiverImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor) for processing images).

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, PerceiverForImageClassificationLearned
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
>>> model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")

>>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 1000]

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: tabby, tabby cat
```

## PerceiverForImageClassificationFourier

### class transformers.PerceiverForImageClassificationFourier

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1264)

( config  )

Parameters

* **config** ([PerceiverForImageClassificationFourier](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationFourier)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses fixed 2D Fourier position embeddings. As shown in the paper, this model can achieve a top-1 accuracy of
79.0 on ImageNet, and 84.5 when pre-trained on a large-scale dataset (i.e. JFT).

[PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) uses [PerceiverImagePreprocessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor)
(with `prep_type="pixels"`) to preprocess the input images, and
[PerceiverClassificationDecoder](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder) to decode the latent representation of
[PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) into classification logits.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1296)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor). See [PerceiverImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor) for processing images).

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForImageClassificationFourier](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationFourier) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, PerceiverForImageClassificationFourier
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-fourier")
>>> model = PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier")

>>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 1000]

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: tabby, tabby cat
```

## PerceiverForImageClassificationConvProcessing

### class transformers.PerceiverForImageClassificationConvProcessing

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1405)

( config  )

Parameters

* **config** ([PerceiverForImageClassificationConvProcessing](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationConvProcessing)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses a 2D conv+maxpool preprocessing network. As shown in the paper, this model can achieve a top-1 accuracy
of 82.1 on ImageNet.

[PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) uses [PerceiverImagePreprocessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor)
(with `prep_type="conv"`) to preprocess the input images, and
[PerceiverClassificationDecoder](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder) to decode the latent representation of
[PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) into classification logits.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1438)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.Tensor] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor). See [PerceiverImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor) for processing images).

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForImageClassificationConvProcessing](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationConvProcessing) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, PerceiverForImageClassificationConvProcessing
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-conv")
>>> model = PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")

>>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 1000]

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: tabby, tabby cat
```

## PerceiverForOpticalFlow

### class transformers.PerceiverForOpticalFlow

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1547)

( config  )

Parameters

* **config** ([PerceiverForOpticalFlow](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForOpticalFlow)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for optical flow, for tasks such as Sintel and KITTI. [PerceiverForOpticalFlow](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForOpticalFlow) uses
[PerceiverImagePreprocessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor) (with *prep\_type=“patches”*) to preprocess the
input images, and [PerceiverOpticalFlowDecoder](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder) to decode the latent
representation of [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel).

As input, one concatenates 2 subsequent frames along the channel dimension and extract a 3 x 3 patch around each pixel
(leading to 3 x 3 x 3 x 2 = 54 values for each pixel). Fixed Fourier position encodings are used to encode the position
of each pixel in the patch. Next, one applies the Perceiver encoder. To decode, one queries the latent representation
using the same encoding used for the input.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1597)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the optical flow loss. Indices should be in `[0, ..., config.num_labels - 1]`.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForOpticalFlow](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForOpticalFlow) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import PerceiverForOpticalFlow
>>> import torch

>>> model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")

>>> # in the Perceiver IO paper, the authors extract a 3 x 3 patch around each pixel,
>>> # leading to 3 x 3 x 3 = 27 values for each pixel (as each pixel also has 3 color channels)
>>> # patches have shape (batch_size, num_frames, num_channels, height, width)
>>> # the authors train on resolutions of 368 x 496
>>> patches = torch.randn(1, 2, 27, 368, 496)
>>> outputs = model(inputs=patches)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 368, 496, 2]
```

## PerceiverForMultimodalAutoencoding

### class transformers.PerceiverForMultimodalAutoencoding

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1690)

( config: PerceiverConfig  )

Parameters

* **config** ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for multimodal (video) autoencoding, for tasks such as Kinetics-700.

[PerceiverForMultimodalAutoencoding](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForMultimodalAutoencoding) uses [PerceiverMultimodalPreprocessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor) to
preprocess the 3 modalities: images, audio and class labels. This preprocessor uses modality-specific preprocessors to
preprocess every modality separately, after which they are concatenated. Trainable position embeddings are used to pad
each modality to the same number of channels to make concatenation along the time dimension possible. Next, one applies
the Perceiver encoder.

[PerceiverMultimodalDecoder](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder) is used to decode the latent representation of
[PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel). This decoder uses each modality-specific decoder to construct queries. The decoder queries are
created based on the inputs after preprocessing. However, autoencoding an entire video in a single forward pass is
computationally infeasible, hence one only uses parts of the decoder queries to do cross-attention with the latent
representation. This is determined by the subsampled indices for each modality, which can be provided as additional
input to the forward pass of [PerceiverForMultimodalAutoencoding](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForMultimodalAutoencoding).

[PerceiverMultimodalDecoder](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder) also pads the decoder queries of the different
modalities to the same number of channels, in order to concatenate them along the time dimension. Next, cross-attention
is performed with the latent representation of [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel).

Finally, `~models.perceiver.modeling_perceiver.PerceiverMultiModalPostprocessor` is used to turn this tensor into an
actual video. It first splits up the output into the different modalities, and then applies the respective
postprocessor for each modality.

Note that, by masking the classification label during evaluation (i.e. simply providing a tensor of zeros for the
“label” modality), this auto-encoding model becomes a Kinetics 700 video classifier.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perceiver/modeling_perceiver.py#L1806)

( inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None subsampled\_output\_points: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **inputs** (`torch.Tensor`, *optional*) —
  Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **subsampled\_output\_points** (`dict[str, torch.Tensor]`, *optional*) —
  Dictionary of tensors used as queries for the decoder. The decoder maps these queries to the latent
  representation of the model. Used for subsampled decoding, e.g. when only decoding certain image patches.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [PerceiverForMultimodalAutoencoding](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForMultimodalAutoencoding) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import PerceiverForMultimodalAutoencoding
>>> import torch
>>> import numpy as np

>>> # create multimodal inputs
>>> images = torch.randn((1, 16, 3, 224, 224))
>>> audio = torch.randn((1, 30720, 1))
>>> inputs = dict(image=images, audio=audio, label=torch.zeros((images.shape[0], 700)))

>>> model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver")

>>> # in the Perceiver IO paper, videos are auto-encoded in chunks
>>> # each chunk subsamples different index dimensions of the image and audio modality decoder queries
>>> nchunks = 128
>>> image_chunk_size = np.prod((16, 224, 224)) // nchunks
>>> audio_chunk_size = audio.shape[1] // model.config.samples_per_patch // nchunks
>>> # process the first chunk
>>> chunk_idx = 0
>>> subsampling = {
...     "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
...     "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
...     "label": None,
... }

>>> outputs = model(inputs=inputs, subsampled_output_points=subsampling)
>>> logits = outputs.logits
>>> list(logits["audio"].shape)
[1, 240]

>>> list(logits["image"].shape)
[1, 6272, 3]

>>> list(logits["label"].shape)
[1, 700]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/perceiver.md)
