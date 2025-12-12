# LXMERT

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

- Bounding boxes are not necessary to be used in the visual feature embeddings, any kind of visual-spacial features
  will work.
- Both the language hidden states and the visual hidden states that LXMERT outputs are passed through the
  cross-modality layer, so they contain information from both modalities. To access a modality that only attends to
  itself, select the vision/language hidden states from the first input in the tuple.
- The bidirectional cross-modality encoder attention only returns attention values when the language modality is used
  as the input and the vision modality is used as the context vector. Further, while the cross-modality encoder
  contains self-attention for each respective modality and cross-attention, only the cross attention is returned and
  both self attention outputs are disregarded.

## Resources

- [Question answering task guide](../tasks/question_answering)

## LxmertConfig[[transformers.LxmertConfig]]

#### transformers.LxmertConfig[[transformers.LxmertConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/configuration_lxmert.py#L24)

This is the configuration class to store the configuration of a [LxmertModel](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertModel). It is used
to instantiate a LXMERT model according to the specified arguments, defining the model architecture. Instantiating
a configuration with the defaults will yield a similar configuration to that of the Lxmert
[unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30522) : Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [LxmertModel](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertModel).

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

num_qa_labels (`int`, *optional*, defaults to 9500) : This represents the total number of different question answering (QA) labels there are. If using more than one dataset with QA, the user will need to account for the total number of labels that all of the datasets have in total.

num_object_labels (`int`, *optional*, defaults to 1600) : This represents the total number of semantically unique objects that lxmert will be able to classify a pooled-object feature as belonging too.

num_attr_labels (`int`, *optional*, defaults to 400) : This represents the total number of semantically unique attributes that lxmert will be able to classify a pooled-object feature as possessing.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

type_vocab_size (`int`, *optional*, defaults to 2) : The vocabulary size of the *token_type_ids* passed into [BertModel](/docs/transformers/main/en/model_doc/bert#transformers.BertModel).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

l_layers (`int`, *optional*, defaults to 9) : Number of hidden layers in the Transformer language encoder.

x_layers (`int`, *optional*, defaults to 5) : Number of hidden layers in the Transformer cross modality encoder.

r_layers (`int`, *optional*, defaults to 5) : Number of hidden layers in the Transformer visual encoder.

visual_feat_dim (`int`, *optional*, defaults to 2048) : This represents the last dimension of the pooled-object features used as input for the model, representing the size of each object feature itself.

visual_pos_dim (`int`, *optional*, defaults to 4) : This represents the number of spatial features that are mixed into the visual features. The default is set to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)

visual_loss_normalizer (`float`, *optional*, defaults to 6.67) : This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one decided to train with multiple vision-based loss objectives.

task_matched (`bool`, *optional*, defaults to `True`) : This task is used for sentence-image matching. If the sentence correctly describes the image the label will be 1. If the sentence does not correctly describe the image, the label will be 0.

task_mask_lm (`bool`, *optional*, defaults to `True`) : Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss objective.

task_obj_predict (`bool`, *optional*, defaults to `True`) : Whether or not to add object prediction, attribute prediction and feature regression to the loss objective.

task_qa (`bool`, *optional*, defaults to `True`) : Whether or not to add the question-answering loss to the objective

visual_obj_loss (`bool`, *optional*, defaults to `True`) : Whether or not to calculate the object-prediction loss objective

visual_attr_loss (`bool`, *optional*, defaults to `True`) : Whether or not to calculate the attribute-prediction loss objective

visual_feat_loss (`bool`, *optional*, defaults to `True`) : Whether or not to calculate the feature-regression loss objective

## LxmertTokenizer[[transformers.BertTokenizer]]

#### transformers.BertTokenizer[[transformers.BertTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py#L43)

Construct a BERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

**Parameters:**

vocab (`str` or `dict[str, int]`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.

do_lower_case (`bool`, *optional*, defaults to `False`) : Whether or not to lowercase the input when tokenizing.

unk_token (`str`, *optional*, defaults to `"[UNK]"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `"[SEP]"`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `"[PAD]"`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `"[CLS]"`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `"[MASK]"`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

tokenize_chinese_chars (`bool`, *optional*, defaults to `True`) : Whether or not to tokenize Chinese characters.

strip_accents (`bool`, *optional*) : Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original BERT).

## LxmertTokenizerFast[[transformers.BertTokenizer]]

#### transformers.BertTokenizer[[transformers.BertTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py#L43)

Construct a BERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

**Parameters:**

vocab (`str` or `dict[str, int]`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.

do_lower_case (`bool`, *optional*, defaults to `False`) : Whether or not to lowercase the input when tokenizing.

unk_token (`str`, *optional*, defaults to `"[UNK]"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `"[SEP]"`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `"[PAD]"`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `"[CLS]"`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `"[MASK]"`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

tokenize_chinese_chars (`bool`, *optional*, defaults to `True`) : Whether or not to tokenize Chinese characters.

strip_accents (`bool`, *optional*) : Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original BERT).

## Lxmert specific outputs[[transformers.models.lxmert.modeling_lxmert.LxmertModelOutput]]

#### transformers.models.lxmert.modeling_lxmert.LxmertModelOutput[[transformers.models.lxmert.modeling_lxmert.LxmertModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L52)

Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
encoder")

**Parameters:**

language_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the language encoder.

vision_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the visual encoder.

pooled_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) : Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed by a Linear layer and a Tanh activation function. The Linear

language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.

vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.

language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput[[transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L139)

Output type of [LxmertForPreTraining](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertForPreTraining).

**Parameters:**

loss (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) : Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.

prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

cross_relationship_score (`torch.FloatTensor` of shape `(batch_size, 2)`) : Prediction scores of the textual matching objective (classification) head (scores of True/False continuation before SoftMax).

question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) : Prediction scores of question answering objective (classification).

language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.

vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.

language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput[[transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L97)

Output type of [LxmertForQuestionAnswering](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering).

**Parameters:**

loss (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) : Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.k.

question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*) : Prediction scores of question answering objective (classification).

language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.

vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.

language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## LxmertModel[[transformers.LxmertModel]]

#### transformers.LxmertModel[[transformers.LxmertModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L686)

The bare Lxmert Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LxmertModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L701[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "visual_feats", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "visual_pos", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "visual_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **visual_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) --
  This input represents visual features. They ROI pooled object features from bounding boxes using a
  faster-RCNN model)

  These are currently not provided by the transformers library.
- **visual_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) --
  This input represents spatial features corresponding to their relative (via index) visual features. The
  pre-trained LXMERT model expects these spatial features to be normalized bounding boxes on a scale of 0 to
  1.

  These are currently not provided by the transformers library.
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **visual_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.lxmert.modeling_lxmert.LxmertModelOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or `tuple(torch.FloatTensor)`A [transformers.models.lxmert.modeling_lxmert.LxmertModelOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

- **language_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the language encoder.
- **vision_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the visual encoder.
- **pooled_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
  by a Linear layer and a Tanh activation function. The Linear
- **language_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **language_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **cross_encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
The [LxmertModel](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([LxmertModel](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.lxmert.modeling_lxmert.LxmertModelOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.lxmert.modeling_lxmert.LxmertModelOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

- **language_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the language encoder.
- **vision_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the visual encoder.
- **pooled_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
  by a Linear layer and a Tanh activation function. The Linear
- **language_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **language_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **cross_encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

## LxmertForPreTraining[[transformers.LxmertForPreTraining]]

#### transformers.LxmertForPreTraining[[transformers.LxmertForPreTraining]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L837)

The Lxmert Model with a specified pretraining head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LxmertForPreTraining.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L987[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "visual_feats", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "visual_pos", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "visual_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "obj_labels", "val": ": typing.Optional[dict[str, tuple[torch.FloatTensor, torch.FloatTensor]]] = None"}, {"name": "matched_label", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "ans", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **visual_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) --
  This input represents visual features. They ROI pooled object features from bounding boxes using a
  faster-RCNN model)

  These are currently not provided by the transformers library.
- **visual_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) --
  This input represents spatial features corresponding to their relative (via index) visual features. The
  pre-trained LXMERT model expects these spatial features to be normalized bounding boxes on a scale of 0 to
  1.

  These are currently not provided by the transformers library.
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **visual_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
  config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
- **obj_labels** (`dict[Str -- tuple[Torch.FloatTensor, Torch.FloatTensor]]`, *optional*):
  each key is named after each one of the visual losses and each element of the tuple is of the shape
  `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and
  the label score respectively
- **matched_label** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the whether or not the text input matches the image (classification) loss. Input
  should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

  - 0 indicates that the sentence does not match the image,
  - 1 indicates that the sentence does match the image.
- **ans** (`Torch.Tensor` of shape `(batch_size)`, *optional*) --
  a one hot representation hof the correct answer *optional*
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or `tuple(torch.FloatTensor)`A [transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

- **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
- **prediction_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **cross_relationship_score** (`torch.FloatTensor` of shape `(batch_size, 2)`) -- Prediction scores of the textual matching objective (classification) head (scores of True/False
  continuation before SoftMax).
- **question_answering_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) -- Prediction scores of question answering objective (classification).
- **language_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **language_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **cross_encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
The [LxmertForPreTraining](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([LxmertForPreTraining](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertForPreTraining)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

- **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
- **prediction_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **cross_relationship_score** (`torch.FloatTensor` of shape `(batch_size, 2)`) -- Prediction scores of the textual matching objective (classification) head (scores of True/False
  continuation before SoftMax).
- **question_answering_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) -- Prediction scores of question answering objective (classification).
- **language_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **language_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **cross_encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

## LxmertForQuestionAnswering[[transformers.LxmertForQuestionAnswering]]

#### transformers.LxmertForQuestionAnswering[[transformers.LxmertForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L1142)

Lxmert Model with a visual-answering head on top for downstream QA tasks

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LxmertForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/lxmert/modeling_lxmert.py#L1234[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "visual_feats", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "visual_pos", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "visual_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **visual_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) --
  This input represents visual features. They ROI pooled object features from bounding boxes using a
  faster-RCNN model)

  These are currently not provided by the transformers library.
- **visual_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) --
  This input represents spatial features corresponding to their relative (via index) visual features. The
  pre-trained LXMERT model expects these spatial features to be normalized bounding boxes on a scale of 0 to
  1.

  These are currently not provided by the transformers library.
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **visual_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`Torch.Tensor` of shape `(batch_size)`, *optional*) --
  A one-hot representation of the correct answer
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`A [transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

- **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.k.
- **question_answering_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*) -- Prediction scores of question answering objective (classification).
- **language_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **language_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **cross_encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
The [LxmertForQuestionAnswering](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([LxmertForQuestionAnswering](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LxmertConfig](/docs/transformers/main/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

- **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.k.
- **question_answering_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, *optional*) -- Prediction scores of question answering objective (classification).
- **language_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
  shape `(batch_size, sequence_length, hidden_size)`.
- **language_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
- **cross_encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
