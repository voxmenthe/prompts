*This model was released on 2019-08-20 and added to Hugging Face Transformers on 2020-11-16.*

# LXMERT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://huggingface.co/papers/1908.07490) by Hao Tan & Mohit Bansal. It is a series of bidirectional transformer encoders
(one for the vision modality, one for the language modality, and then one to fuse both modalities) pretrained using a
combination of masked language modeling, visual-language text alignment, ROI-feature regression, masked
visual-attribute modeling, masked visual-object modeling, and visual-question answering objectives. The pretraining
consists of multiple multi-modal datasets: MSCOCO, Visual-Genome + Visual-Genome Question Answering, VQA 2.0, and GQA.

The abstract from the paper is the following:

*Vision-and-language reasoning requires an understanding of visual concepts, language semantics, and, most importantly,
the alignment and relationships between these two modalities. We thus propose the LXMERT (Learning Cross-Modality
Encoder Representations from Transformers) framework to learn these vision-and-language connections. In LXMERT, we
build a large-scale Transformer model that consists of three encoders: an object relationship encoder, a language
encoder, and a cross-modality encoder. Next, to endow our model with the capability of connecting vision and language
semantics, we pre-train the model with large amounts of image-and-sentence pairs, via five diverse representative
pretraining tasks: masked language modeling, masked object prediction (feature regression and label classification),
cross-modality matching, and image question answering. These tasks help in learning both intra-modality and
cross-modality relationships. After fine-tuning from our pretrained parameters, our model achieves the state-of-the-art
results on two visual question answering datasets (i.e., VQA and GQA). We also show the generalizability of our
pretrained cross-modality model by adapting it to a challenging visual-reasoning task, NLVR, and improve the previous
best result by 22% absolute (54% to 76%). Lastly, we demonstrate detailed ablation studies to prove that both our novel
model components and pretraining strategies significantly contribute to our strong results; and also present several
attention visualizations for the different encoders*

This model was contributed by [eltoto1219](https://huggingface.co/eltoto1219). The original code can be found [here](https://github.com/airsplay/lxmert).

## Usage tips

* Bounding boxes are not necessary to be used in the visual feature embeddings, any kind of visual-spacial features
  will work.
* Both the language hidden states and the visual hidden states that LXMERT outputs are passed through the
  cross-modality layer, so they contain information from both modalities. To access a modality that only attends to
  itself, select the vision/language hidden states from the first input in the tuple.
* The bidirectional cross-modality encoder attention only returns attention values when the language modality is used
  as the input and the vision modality is used as the context vector. Further, while the cross-modality encoder
  contains self-attention for each respective modality and cross-attention, only the cross attention is returned and
  both self attention outputs are disregarded.

## Resources

* [Question answering task guide](../tasks/question_answering)

## LxmertConfig

### class transformers.LxmertConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/configuration_lxmert.py#L24)

( vocab\_size = 30522 hidden\_size = 768 num\_attention\_heads = 12 num\_qa\_labels = 9500 num\_object\_labels = 1600 num\_attr\_labels = 400 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 l\_layers = 9 x\_layers = 5 r\_layers = 5 visual\_feat\_dim = 2048 visual\_pos\_dim = 4 visual\_loss\_normalizer = 6.67 task\_matched = True task\_mask\_lm = True task\_obj\_predict = True task\_qa = True visual\_obj\_loss = True visual\_attr\_loss = True visual\_feat\_loss = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [LxmertModel](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertModel) or `TFLxmertModel`.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_qa\_labels** (`int`, *optional*, defaults to 9500) —
  This represents the total number of different question answering (QA) labels there are. If using more than
  one dataset with QA, the user will need to account for the total number of labels that all of the datasets
  have in total.
* **num\_object\_labels** (`int`, *optional*, defaults to 1600) —
  This represents the total number of semantically unique objects that lxmert will be able to classify a
  pooled-object feature as belonging too.
* **num\_attr\_labels** (`int`, *optional*, defaults to 400) —
  This represents the total number of semantically unique attributes that lxmert will be able to classify a
  pooled-object feature as possessing.
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
  The vocabulary size of the *token\_type\_ids* passed into [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **l\_layers** (`int`, *optional*, defaults to 9) —
  Number of hidden layers in the Transformer language encoder.
* **x\_layers** (`int`, *optional*, defaults to 5) —
  Number of hidden layers in the Transformer cross modality encoder.
* **r\_layers** (`int`, *optional*, defaults to 5) —
  Number of hidden layers in the Transformer visual encoder.
* **visual\_feat\_dim** (`int`, *optional*, defaults to 2048) —
  This represents the last dimension of the pooled-object features used as input for the model, representing
  the size of each object feature itself.
* **visual\_pos\_dim** (`int`, *optional*, defaults to 4) —
  This represents the number of spatial features that are mixed into the visual features. The default is set
  to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)
* **visual\_loss\_normalizer** (`float`, *optional*, defaults to 6.67) —
  This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one
  decided to train with multiple vision-based loss objectives.
* **task\_matched** (`bool`, *optional*, defaults to `True`) —
  This task is used for sentence-image matching. If the sentence correctly describes the image the label will
  be 1. If the sentence does not correctly describe the image, the label will be 0.
* **task\_mask\_lm** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss
  objective.
* **task\_obj\_predict** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add object prediction, attribute prediction and feature regression to the loss objective.
* **task\_qa** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add the question-answering loss to the objective
* **visual\_obj\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether or not to calculate the object-prediction loss objective
* **visual\_attr\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether or not to calculate the attribute-prediction loss objective
* **visual\_feat\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether or not to calculate the feature-regression loss objective

This is the configuration class to store the configuration of a [LxmertModel](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertModel) or a `TFLxmertModel`. It is used
to instantiate a LXMERT model according to the specified arguments, defining the model architecture. Instantiating
a configuration with the defaults will yield a similar configuration to that of the Lxmert
[unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## LxmertTokenizer

### class transformers.LxmertTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/tokenization_lxmert.py#L53)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None clean\_up\_tokenization\_spaces = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **do\_basic\_tokenize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to do basic tokenization before WordPiece.
* **never\_split** (`Iterable`, *optional*) —
  Collection of tokens which will never be split during tokenization. Only has an effect when
  `do_basic_tokenize=True`
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **tokenize\_chinese\_chars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to tokenize Chinese characters.

  This should likely be deactivated for Japanese (see this
  [issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original Lxmert).
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `True`) —
  Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
  extra spaces.

Construct a Lxmert tokenizer. Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/tokenization_lxmert.py#L188)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Lxmert sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/tokenization_lxmert.py#L183)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/tokenization_lxmert.py#L213)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`List[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

## LxmertTokenizerFast

### class transformers.LxmertTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/tokenization_lxmert_fast.py#L29)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **clean\_text** (`bool`, *optional*, defaults to `True`) —
  Whether or not to clean the text before tokenization by removing any control characters and replacing all
  whitespaces by the classic one.
* **tokenize\_chinese\_chars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
  issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original Lxmert).
* **wordpieces\_prefix** (`str`, *optional*, defaults to `"##"`) —
  The prefix for subwords.

Construct a “fast” Lxmert tokenizer (backed by HuggingFace’s *tokenizers* library). Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/tokenization_lxmert_fast.py#L114)

( token\_ids\_0 token\_ids\_1 = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Lxmert sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

## Lxmert specific outputs

### class transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L52)

( language\_output: typing.Optional[torch.FloatTensor] = None vision\_output: typing.Optional[torch.FloatTensor] = None pooled\_output: typing.Optional[torch.FloatTensor] = None language\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None vision\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None language\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None vision\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **language\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Sequence of hidden-states at the output of the last layer of the language encoder.
* **vision\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Sequence of hidden-states at the output of the last layer of the visual encoder.
* **pooled\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) —
  Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
  by a Linear layer and a Tanh activation function. The Linear
* **language\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **language\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

Lxmert’s outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the “relation-ship”
encoder”)

### class transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L139)

( loss: typing.Optional[torch.FloatTensor] = None prediction\_logits: typing.Optional[torch.FloatTensor] = None cross\_relationship\_score: typing.Optional[torch.FloatTensor] = None question\_answering\_score: typing.Optional[torch.FloatTensor] = None language\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None vision\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None language\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None vision\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) —
  Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
* **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) —
  Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **cross\_relationship\_score** (`torch.FloatTensor` of shape `(batch_size, 2)`) —
  Prediction scores of the textual matching objective (classification) head (scores of True/False
  continuation before SoftMax).
* **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) —
  Prediction scores of question answering objective (classification).
* **language\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **language\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

Output type of [LxmertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForPreTraining).

### class transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L97)

( loss: typing.Optional[torch.FloatTensor] = None question\_answering\_score: typing.Optional[torch.FloatTensor] = None language\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None vision\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None language\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None vision\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) —
  Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.k.
* **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*) —
  Prediction scores of question answering objective (classification).
* **language\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **language\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

Output type of [LxmertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering).

## LxmertModel

### class transformers.LxmertModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L787)

( config  )

Parameters

* **config** ([LxmertModel](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Lxmert Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L802)

( input\_ids: typing.Optional[torch.LongTensor] = None visual\_feats: typing.Optional[torch.FloatTensor] = None visual\_pos: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **visual\_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) —
  This input represents visual features. They ROI pooled object features from bounding boxes using a
  faster-RCNN model)

  These are currently not provided by the transformers library.
* **visual\_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) —
  This input represents spatial features corresponding to their relative (via index) visual features. The
  pre-trained LXMERT model expects these spatial features to be normalized bounding boxes on a scale of 0 to


  These are currently not provided by the transformers library.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

* **language\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the language encoder.
* **vision\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the visual encoder.
* **pooled\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
  by a Linear layer and a Tanh activation function. The Linear
* **language\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **language\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

The [LxmertModel](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## LxmertForPreTraining

### class transformers.LxmertForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L937)

( config  )

Parameters

* **config** ([LxmertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForPreTraining)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Lxmert Model with a specified pretraining head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L1087)

( input\_ids: typing.Optional[torch.LongTensor] = None visual\_feats: typing.Optional[torch.FloatTensor] = None visual\_pos: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None obj\_labels: typing.Optional[dict[str, tuple[torch.FloatTensor, torch.FloatTensor]]] = None matched\_label: typing.Optional[torch.LongTensor] = None ans: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **visual\_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) —
  This input represents visual features. They ROI pooled object features from bounding boxes using a
  faster-RCNN model)

  These are currently not provided by the transformers library.
* **visual\_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) —
  This input represents spatial features corresponding to their relative (via index) visual features. The
  pre-trained LXMERT model expects these spatial features to be normalized bounding boxes on a scale of 0 to


  These are currently not provided by the transformers library.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **obj\_labels** (`dict[Str -- tuple[Torch.FloatTensor, Torch.FloatTensor]]`, *optional*):
  each key is named after each one of the visual losses and each element of the tuple is of the shape
  `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and
  the label score respectively
* **matched\_label** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the whether or not the text input matches the image (classification) loss. Input
  should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:
  + 0 indicates that the sentence does not match the image,
  + 1 indicates that the sentence does match the image.
* **ans** (`Torch.Tensor` of shape `(batch_size)`, *optional*) —
  a one hot representation hof the correct answer *optional*
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
* **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **cross\_relationship\_score** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the textual matching objective (classification) head (scores of True/False
  continuation before SoftMax).
* **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) — Prediction scores of question answering objective (classification).
* **language\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **language\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

The [LxmertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## LxmertForQuestionAnswering

### class transformers.LxmertForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L1242)

( config  )

Parameters

* **config** ([LxmertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Lxmert Model with a visual-answering head on top for downstream QA tasks

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lxmert/modeling_lxmert.py#L1334)

( input\_ids: typing.Optional[torch.LongTensor] = None visual\_feats: typing.Optional[torch.FloatTensor] = None visual\_pos: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **visual\_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) —
  This input represents visual features. They ROI pooled object features from bounding boxes using a
  faster-RCNN model)

  These are currently not provided by the transformers library.
* **visual\_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) —
  This input represents spatial features corresponding to their relative (via index) visual features. The
  pre-trained LXMERT model expects these spatial features to be normalized bounding boxes on a scale of 0 to


  These are currently not provided by the transformers library.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`Torch.Tensor` of shape `(batch_size)`, *optional*) —
  A one-hot representation of the correct answer
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.k.
* **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*) — Prediction scores of question answering objective (classification).
* **language\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
* **language\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
* **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

The [LxmertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, LxmertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
>>> model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/lxmert.md)
