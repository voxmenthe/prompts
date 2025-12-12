# XLNet

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

- The specific attention pattern can be controlled at training and test time using the `perm_mask` input.
- Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained
  using only a sub-set of the output tokens as target which are selected with the `target_mapping` input.
- To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
  `target_mapping` inputs to control the attention span and outputs (see examples in
  *examples/pytorch/text-generation/run_generation.py*)
- XLNet is one of the few models that has no sequence length limit.
- XLNet is not a traditional autoregressive model but uses a training strategy that builds on that. It permutes the tokens in the sentence, then allows the model to use the last n tokens to predict the token n+1. Since this is all done with a mask, the sentence is actually fed in the model in the right order, but instead of masking the first n tokens for n+1, XLNet uses a mask that hides the previous tokens in some given permutation of 1,…,sequence length.
- XLNet also uses the same recurrence mechanism as Transformer-XL to build long-term dependencies.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## XLNetConfig[[transformers.XLNetConfig]]

#### transformers.XLNetConfig[[transformers.XLNetConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/configuration_xlnet.py#L27)

This is the configuration class to store the configuration of a [XLNetModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetModel). It is used to
instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
[xlnet/xlnet-large-cased](https://huggingface.co/xlnet/xlnet-large-cased) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import XLNetConfig, XLNetModel

>>> # Initializing a XLNet configuration
>>> configuration = XLNetConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = XLNetModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 32000) : Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [XLNetModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetModel).

d_model (`int`, *optional*, defaults to 1024) : Dimensionality of the encoder layers and the pooler layer.

n_layer (`int`, *optional*, defaults to 24) : Number of hidden layers in the Transformer encoder.

n_head (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

d_inner (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.

ff_activation (`str` or `Callable`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

attn_type (`str`, *optional*, defaults to `"bi"`) : The attention type used by the model. Set `"bi"` for XLNet, `"uni"` for Transformer-XL.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

mem_len (`int` or `None`, *optional*) : The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous forward pass won't be re-computed. See the [quickstart](https://huggingface.co/transformers/quickstart.html#using-the-past) for more information.

reuse_len (`int`, *optional*) : The number of tokens in the current batch to be cached and reused in the future.

bi_data (`bool`, *optional*, defaults to `False`) : Whether or not to use bidirectional input pipeline. Usually set to `True` during pretraining and `False` during finetuning.

clamp_len (`int`, *optional*, defaults to -1) : Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.

same_length (`bool`, *optional*, defaults to `False`) : Whether or not to use the same attention length for each token.

summary_type (`str`, *optional*, defaults to "last") : Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.  Has to be one of the following options:  - `"last"`: Take the last token hidden state (like XLNet). - `"first"`: Take the first token hidden state (like BERT). - `"mean"`: Take the mean of all tokens hidden states. - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2). - `"attn"`: Not implemented now, use multi-head attention.

summary_use_proj (`bool`, *optional*, defaults to `True`) : Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.  Whether or not to add a projection after the vector extraction.

summary_activation (`str`, *optional*) : Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.  Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.

summary_proj_to_labels (`boo`, *optional*, defaults to `True`) : Used in the sequence classification and multiple choice models.  Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.

summary_last_dropout (`float`, *optional*, defaults to 0.1) : Used in the sequence classification and multiple choice models.  The dropout ratio to be used after the projection and activation.

start_n_top (`int`, *optional*, defaults to 5) : Used in the SQuAD evaluation script.

end_n_top (`int`, *optional*, defaults to 5) : Used in the SQuAD evaluation script.

use_mems_eval (`bool`, *optional*, defaults to `True`) : Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.

use_mems_train (`bool`, *optional*, defaults to `False`) : Whether or not the model should make use of the recurrent memory mechanism in train mode.    For pretraining, it is recommended to set `use_mems_train` to `True`. For fine-tuning, it is recommended to set `use_mems_train` to `False` as discussed [here](https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587). If `use_mems_train` is set to `True`, one has to make sure that the train batches are correctly pre-processed, *e.g.* `batch_1 = [[This line is], [This is the]]` and `batch_2 = [[ the first line], [ second line]]` and that all batches are of equal size.  

## XLNetTokenizer[[transformers.XLNetTokenizer]]

#### transformers.XLNetTokenizer[[transformers.XLNetTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/tokenization_xlnet.py#L41)

Construct a XLNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

get_special_tokens_masktransformers.XLNetTokenizer.get_special_tokens_maskhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1335[{"name": "token_ids_0", "val": ": list[int]"}, {"name": "token_ids_1", "val": ": Optional[list[int]] = None"}, {"name": "already_has_special_tokens", "val": ": bool = False"}]- **token_ids_0** -- List of IDs for the (possibly already formatted) sequence.
- **token_ids_1** -- Unused when `already_has_special_tokens=True`. Must be None in that case.
- **already_has_special_tokens** -- Whether the sequence is already formatted with special tokens.0A list of integers in the range [0, 1]1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with `already_has_special_tokens=True` to build a mask over an
already-formatted sequence. In that case, we compute the mask by checking membership in `all_special_ids`.

**Parameters:**

vocab (`list of tuples`, *optional*) : List of (token, score) tuples for Unigram model. If not provided, an empty list is used.

unk_id (`int`, *optional*, defaults to 0) : The ID of the unknown token in the vocabulary.

do_lower_case (`bool`, *optional*, defaults to `False`) : Whether to lowercase the input when tokenizing.

remove_space (`bool`, *optional*, defaults to `True`) : Whether to strip the text when tokenizing (removing excess spaces before and after the string).

keep_accents (`bool`, *optional*, defaults to `False`) : Whether to keep accents when tokenizing.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

additional_special_tokens (`list[str]`, *optional*, defaults to `["", ""]`) : Additional special tokens used by the tokenizer.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### save_vocabulary[[transformers.XLNetTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L384)

## XLNetTokenizerFast[[transformers.XLNetTokenizer]]

#### transformers.XLNetTokenizer[[transformers.XLNetTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/tokenization_xlnet.py#L41)

Construct a XLNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

**Parameters:**

vocab (`list of tuples`, *optional*) : List of (token, score) tuples for Unigram model. If not provided, an empty list is used.

unk_id (`int`, *optional*, defaults to 0) : The ID of the unknown token in the vocabulary.

do_lower_case (`bool`, *optional*, defaults to `False`) : Whether to lowercase the input when tokenizing.

remove_space (`bool`, *optional*, defaults to `True`) : Whether to strip the text when tokenizing (removing excess spaces before and after the string).

keep_accents (`bool`, *optional*, defaults to `False`) : Whether to keep accents when tokenizing.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

additional_special_tokens (`list[str]`, *optional*, defaults to `["", ""]`) : Additional special tokens used by the tokenizer.

## XLNet specific outputs[[transformers.models.xlnet.modeling_xlnet.XLNetModelOutput]]

#### transformers.models.xlnet.modeling_xlnet.XLNetModelOutput[[transformers.models.xlnet.modeling_xlnet.XLNetModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L666)

Output type of [XLNetModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetModel).

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) : Sequence of hidden-states at the last layer of the model.  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput[[transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L691)

Output type of [XLNetLMHeadModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetLMHeadModel).

**Parameters:**

loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) : Language modeling loss (for next-token prediction).

logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput[[transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L719)

Output type of [XLNetForSequenceClassification](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForSequenceClassification).

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) : Classification (or regression if config.num_labels==1) scores (before SoftMax).

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput[[transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L769)

Output type of [XLNetForMultipleChoice](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForMultipleChoice).

**Parameters:**

loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) : Classification loss.

logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`) : *num_choices* is the second dimension of the input tensors. (see *input_ids* above).  Classification scores (before SoftMax).

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput[[transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L744)

Output type of `XLNetForTokenClassificationOutput`.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification loss.

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) : Classification scores (before SoftMax).

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput[[transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L796)

Output type of [XLNetForQuestionAnsweringSimple](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple).

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) : Span-start scores (before SoftMax).

end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) : Span-end scores (before SoftMax).

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput[[transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L824)

Output type of [XLNetForQuestionAnswering](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering).

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) : Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.

start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) : Log probabilities for the top config.start_n_top start token possibilities (beam-search).

start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) : Indices for the top config.start_n_top start token possibilities (beam-search).

end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) : Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).

end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) : Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).

cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) : Log probabilities for the `is_impossible` label of the answers.

mems (`list[torch.FloatTensor]` of length `config.n_layers`) : Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as `input_ids` as they have already been computed.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## XLNetModel[[transformers.XLNetModel]]

#### transformers.XLNetModel[[transformers.XLNetModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L858)

The bare Xlnet Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L983[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.xlnet.modeling_xlnet.XLNetModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) -- Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([XLNetModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) -- Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLNetLMHeadModel[[transformers.XLNetLMHeadModel]]

#### transformers.XLNetLMHeadModel[[transformers.XLNetLMHeadModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1227)

XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetLMHeadModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1300[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, num_predict)`, *optional*) --
  Labels for masked language modeling. `num_predict` corresponds to `target_mapping.shape[1]`. If
  `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

  The labels should correspond to the masked input words that should be predicted and depends on
  `target_mapping`. Note in order to perform standard auto-regressive language modeling a ** token has
  to be added to the `input_ids` (see the `prepare_inputs_for_generation` function and examples below)

  Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored, the loss
  is only computed for labels in `[0, ..., config.vocab_size]`
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).0[transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetLMHeadModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoTokenizer, XLNetLMHeadModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
>>> model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-large-cased")

>>> # We show how to setup inputs to predict a next token using a bi-directional context.
>>> input_ids = torch.tensor(
...     tokenizer.encode("Hello, my dog is very ", add_special_tokens=False)
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
...     tokenizer.encode("Hello, my dog is very ", add_special_tokens=False)
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

**Parameters:**

config ([XLNetLMHeadModel](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetLMHeadModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLNetForSequenceClassification[[transformers.XLNetForSequenceClassification]]

#### transformers.XLNetForSequenceClassification[[transformers.XLNetForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1474)

XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1487[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetForSequenceClassification](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:

```python
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

```python
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

**Parameters:**

config ([XLNetForSequenceClassification](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLNetForMultipleChoice[[transformers.XLNetForMultipleChoice]]

#### transformers.XLNetForMultipleChoice[[transformers.XLNetForMultipleChoice]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1710)

The Xlnet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetForMultipleChoice.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1721[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, num_choices, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) -- *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

  Classification scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetForMultipleChoice](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([XLNetForMultipleChoice](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForMultipleChoice)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) -- *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

  Classification scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLNetForTokenClassification[[transformers.XLNetForTokenClassification]]

#### transformers.XLNetForTokenClassification[[transformers.XLNetForTokenClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1602)

The Xlnet transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetForTokenClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1613[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
  where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.emory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetForTokenClassification](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([XLNetForTokenClassification](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForTokenClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLNetForQuestionAnsweringSimple[[transformers.XLNetForQuestionAnsweringSimple]]

#### transformers.XLNetForQuestionAnsweringSimple[[transformers.XLNetForQuestionAnsweringSimple]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1854)

XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetForQuestionAnsweringSimple.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1865[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **start_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) -- Span-end scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetForQuestionAnsweringSimple](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([XLNetForQuestionAnsweringSimple](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`) -- Span-end scores (before SoftMax).
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLNetForQuestionAnswering[[transformers.XLNetForQuestionAnswering]]

#### transformers.XLNetForQuestionAnswering[[transformers.XLNetForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1975)

The Xlnet transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XLNetForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py#L1989[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mems", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "perm_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "target_mapping", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "input_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "is_impossible", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cls_index", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "p_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_mems", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) --
  Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential
  decoding. The token ids which have their past given to this model should not be passed as `input_ids` as
  they have already been computed.

  `use_mems` has to be set to `True` to make use of `mems`.
- **perm_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*) --
  Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:

  - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;
  - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.

  If not set, each token attends to all the others (full bidirectional attention). Only used during
  pretraining (to define factorization order) or for sequential decoding (generation).
- **target_mapping** (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*) --
  Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is
  on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
  (generation).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **input_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, *optional*) --
  Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for
  real tokens and 1 for padding which is kept for compatibility with the original code base.

  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **masked**,
  - 0 for tokens that are **not masked**.

  You can only uses one of `input_mask` and `attention_mask`.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **start_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **is_impossible** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels whether a question has an answer or no answer (SQuAD 2.0)
- **cls_index** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the classification token to use as input for computing plausibility of the
  answer.
- **p_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
  masked. 0.0 mean token is not masked.
- **use_mems** (`bool`, *optional*) --
  Whether to use memory states to speed up sequential decoding. If set to `True`, the model will use the hidden
  states from previous forward passes to compute attention, which can significantly improve performance for
  sequential decoding tasks.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`A [transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) -- Classification loss as the sum of start token, end token (and is_impossible if provided) classification
  losses.
- **start_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top config.start_n_top start token possibilities (beam-search).
- **start_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top config.start_n_top start token possibilities (beam-search).
- **end_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
- **end_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
- **cls_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the `is_impossible` label of the answers.
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLNetForQuestionAnswering](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([XLNetForQuestionAnswering](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput](/docs/transformers/main/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLNetConfig](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) -- Classification loss as the sum of start token, end token (and is_impossible if provided) classification
  losses.
- **start_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top config.start_n_top start token possibilities (beam-search).
- **start_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top config.start_n_top start token possibilities (beam-search).
- **end_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
- **end_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
- **cls_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the `is_impossible` label of the answers.
- **mems** (`list[torch.FloatTensor]` of length `config.n_layers`) -- Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
  token ids which have their past given to this model should not be passed as `input_ids` as they have
  already been computed.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
