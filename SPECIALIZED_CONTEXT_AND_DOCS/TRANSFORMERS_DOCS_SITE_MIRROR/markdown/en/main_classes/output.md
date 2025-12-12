# Model outputs

All models have outputs that are instances of subclasses of [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput). Those are
data structures containing all the information returned by the model, but that can also be used as tuples or
dictionaries.

Let's see how this looks in an example:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
```

The `outputs` object is a [SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput), as we can see in the
documentation of that class below, it means it has an optional `loss`, a `logits`, an optional `hidden_states` and
an optional `attentions` attribute. Here we have the `loss` since we passed along `labels`, but we don't have
`hidden_states` and `attentions` because we didn't pass `output_hidden_states=True` or
`output_attentions=True`.

When passing `output_hidden_states=True` you may expect the `outputs.hidden_states[-1]` to match `outputs.last_hidden_state` exactly.
However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it's returned.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get `None`. Here for instance `outputs.loss` is the loss computed by the model, and `outputs.attentions` is
`None`.

When considering our `outputs` object as tuple, it only considers the attributes that don't have `None` values.
Here for instance, it has two elements, `loss` then `logits`, so

```python
outputs[:2]
```

will return the tuple `(outputs.loss, outputs.logits)` for instance.

When considering our `outputs` object as dictionary, it only considers the attributes that don't have `None`
values. Here for instance, it has two keys that are `loss` and `logits`.

We document here the generic model outputs that are used by more than one model type. Specific output types are
documented on their corresponding model page.

## ModelOutput[[transformers.utils.ModelOutput]]

#### transformers.utils.ModelOutput[[transformers.utils.ModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L263)

Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
python dictionary.

You can't unpack a `ModelOutput` directly. Use the [to_tuple()](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput.to_tuple) method to convert it to a tuple
before.

to_tupletransformers.utils.ModelOutput.to_tuplehttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L400[]

Convert self to a tuple containing all the attributes/keys that are not `None`.

## BaseModelOutput[[transformers.modeling_outputs.BaseModelOutput]]

#### transformers.modeling_outputs.BaseModelOutput[[transformers.modeling_outputs.BaseModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L26)

Base class for model's outputs, with potential hidden states and attentions.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## BaseModelOutputWithPooling[[transformers.modeling_outputs.BaseModelOutputWithPooling]]

#### transformers.modeling_outputs.BaseModelOutputWithPooling[[transformers.modeling_outputs.BaseModelOutputWithPooling]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L71)

Base class for model's outputs that also contains a pooling of the last hidden states.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) : Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## BaseModelOutputWithCrossAttentions[[transformers.modeling_outputs.BaseModelOutputWithCrossAttentions]]

#### transformers.modeling_outputs.BaseModelOutputWithCrossAttentions[[transformers.modeling_outputs.BaseModelOutputWithCrossAttentions]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L161)

Base class for model's outputs, with potential hidden states and attentions.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

## BaseModelOutputWithPoolingAndCrossAttentions[[transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions]]

#### transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions[[transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L194)

Base class for model's outputs that also contains a pooling of the last hidden states.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) : Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

## BaseModelOutputWithPast[[transformers.modeling_outputs.BaseModelOutputWithPast]]

#### transformers.modeling_outputs.BaseModelOutputWithPast[[transformers.modeling_outputs.BaseModelOutputWithPast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L125)

Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## BaseModelOutputWithPastAndCrossAttentions[[transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions]]

#### transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions[[transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L240)

Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

## Seq2SeqModelOutput[[transformers.modeling_outputs.Seq2SeqModelOutput]]

#### transformers.modeling_outputs.Seq2SeqModelOutput[[transformers.modeling_outputs.Seq2SeqModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L502)

Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
decoding.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the decoder of the model.  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

## CausalLMOutput[[transformers.modeling_outputs.CausalLMOutput]]

#### transformers.modeling_outputs.CausalLMOutput[[transformers.modeling_outputs.CausalLMOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L631)

Base class for causal language model (or autoregressive) outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Language modeling loss (for next-token prediction).

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## CausalLMOutputWithCrossAttentions[[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions]]

#### transformers.modeling_outputs.CausalLMOutputWithCrossAttentions[[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L695)

Base class for causal language model (or autoregressive) outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Language modeling loss (for next-token prediction).

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

## CausalLMOutputWithPast[[transformers.modeling_outputs.CausalLMOutputWithPast]]

#### transformers.modeling_outputs.CausalLMOutputWithPast[[transformers.modeling_outputs.CausalLMOutputWithPast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L660)

Base class for causal language model (or autoregressive) outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Language modeling loss (for next-token prediction).

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## MaskedLMOutput[[transformers.modeling_outputs.MaskedLMOutput]]

#### transformers.modeling_outputs.MaskedLMOutput[[transformers.modeling_outputs.MaskedLMOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L772)

Base class for masked language models outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Masked language modeling (MLM) loss.

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## Seq2SeqLMOutput[[transformers.modeling_outputs.Seq2SeqLMOutput]]

#### transformers.modeling_outputs.Seq2SeqLMOutput[[transformers.modeling_outputs.Seq2SeqLMOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L801)

Base class for sequence-to-sequence language models outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Language modeling loss.

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

## NextSentencePredictorOutput[[transformers.modeling_outputs.NextSentencePredictorOutput]]

#### transformers.modeling_outputs.NextSentencePredictorOutput[[transformers.modeling_outputs.NextSentencePredictorOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L932)

Base class for outputs of models predicting if two sentences are consecutive or not.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `next_sentence_label` is provided) : Next sequence prediction (classification) loss.

logits (`torch.FloatTensor` of shape `(batch_size, 2)`) : Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## SequenceClassifierOutput[[transformers.modeling_outputs.SequenceClassifierOutput]]

#### transformers.modeling_outputs.SequenceClassifierOutput[[transformers.modeling_outputs.SequenceClassifierOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L962)

Base class for outputs of sentence classification models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) : Classification (or regression if config.num_labels==1) scores (before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## Seq2SeqSequenceClassifierOutput[[transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput]]

#### transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput[[transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L991)

Base class for outputs of sequence-to-sequence sentence classification models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) : Classification (or regression if config.num_labels==1) scores (before SoftMax).

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

## MultipleChoiceModelOutput[[transformers.modeling_outputs.MultipleChoiceModelOutput]]

#### transformers.modeling_outputs.MultipleChoiceModelOutput[[transformers.modeling_outputs.MultipleChoiceModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1049)

Base class for outputs of multiple choice models.

**Parameters:**

loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) : Classification loss.

logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`) : *num_choices* is the second dimension of the input tensors. (see *input_ids* above).  Classification scores (before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## TokenClassifierOutput[[transformers.modeling_outputs.TokenClassifierOutput]]

#### transformers.modeling_outputs.TokenClassifierOutput[[transformers.modeling_outputs.TokenClassifierOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1080)

Base class for outputs of token classification models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification loss.

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) : Classification scores (before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## QuestionAnsweringModelOutput[[transformers.modeling_outputs.QuestionAnsweringModelOutput]]

#### transformers.modeling_outputs.QuestionAnsweringModelOutput[[transformers.modeling_outputs.QuestionAnsweringModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1109)

Base class for outputs of question answering models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) : Span-start scores (before SoftMax).

end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) : Span-end scores (before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## Seq2SeqQuestionAnsweringModelOutput[[transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput]]

#### transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput[[transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1141)

Base class for outputs of sequence-to-sequence question answering models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) : Span-start scores (before SoftMax).

end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) : Span-end scores (before SoftMax).

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

## Seq2SeqSpectrogramOutput[[transformers.modeling_outputs.Seq2SeqSpectrogramOutput]]

#### transformers.modeling_outputs.Seq2SeqSpectrogramOutput[[transformers.modeling_outputs.Seq2SeqSpectrogramOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1472)

Base class for sequence-to-sequence spectrogram outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Spectrogram generation loss.

spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) : The predicted spectrogram.

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

## SemanticSegmenterOutput[[transformers.modeling_outputs.SemanticSegmenterOutput]]

#### transformers.modeling_outputs.SemanticSegmenterOutput[[transformers.modeling_outputs.SemanticSegmenterOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1202)

Base class for outputs of semantic segmentation models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) : Classification scores for each pixel.    The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the original image size as post-processing. You should always check your logits shape and resize as needed.   

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## ImageClassifierOutput[[transformers.modeling_outputs.ImageClassifierOutput]]

#### transformers.modeling_outputs.ImageClassifierOutput[[transformers.modeling_outputs.ImageClassifierOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1240)

Base class for outputs of image classification models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) : Classification (or regression if config.num_labels==1) scores (before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the model at the output of each stage.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## ImageClassifierOutputWithNoAttention[[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention]]

#### transformers.modeling_outputs.ImageClassifierOutputWithNoAttention[[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1268)

Base class for outputs of image classification models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) : Classification (or regression if config.num_labels==1) scores (before SoftMax).

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

## DepthEstimatorOutput[[transformers.modeling_outputs.DepthEstimatorOutput]]

#### transformers.modeling_outputs.DepthEstimatorOutput[[transformers.modeling_outputs.DepthEstimatorOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1289)

Base class for outputs of depth estimation models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification (or regression if config.num_labels==1) loss.

predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`) : Predicted depth for each pixel. 

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## Wav2Vec2BaseModelOutput[[transformers.modeling_outputs.Wav2Vec2BaseModelOutput]]

#### transformers.modeling_outputs.Wav2Vec2BaseModelOutput[[transformers.modeling_outputs.Wav2Vec2BaseModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1347)

Base class for models that have been trained with the Wav2Vec2 loss objective.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) : Sequence of extracted feature vectors of the last convolutional layer of the model.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## XVectorOutput[[transformers.modeling_outputs.XVectorOutput]]

#### transformers.modeling_outputs.XVectorOutput[[transformers.modeling_outputs.XVectorOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1376)

Output type of [Wav2Vec2ForXVector](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector).

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Classification loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) : Classification hidden states before AMSoftmax.

embeddings (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) : Utterance embeddings used for vector similarity-based retrieval.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## Seq2SeqTSModelOutput[[transformers.modeling_outputs.Seq2SeqTSModelOutput]]

#### transformers.modeling_outputs.Seq2SeqTSModelOutput[[transformers.modeling_outputs.Seq2SeqTSModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1530)

Base class for time series model's encoder outputs that also contains pre-computed hidden states that can speed up
sequential decoding.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the decoder of the model.  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

loc (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) : Shift values of each time series' context window which is used to give the model inputs of the same magnitude and then used to shift back to the original magnitude.

scale (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) : Scaling values of each time series' context window which is used to give the model inputs of the same magnitude and then used to rescale back to the original magnitude.

static_features (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) : Static features of each time series' in a batch which are copied to the covariates at inference time.

## Seq2SeqTSPredictionOutput[[transformers.modeling_outputs.Seq2SeqTSPredictionOutput]]

#### transformers.modeling_outputs.Seq2SeqTSPredictionOutput[[transformers.modeling_outputs.Seq2SeqTSPredictionOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1600)

Base class for time series model's decoder outputs that also contain the loss as well as the parameters of the
chosen distribution.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when a `future_values` is provided) : Distributional loss.

params (`torch.FloatTensor` of shape `(batch_size, num_samples, num_params)`) : Parameters of the chosen distribution.

past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

loc (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) : Shift values of each time series' context window which is used to give the model inputs of the same magnitude and then used to shift back to the original magnitude.

scale (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) : Scaling values of each time series' context window which is used to give the model inputs of the same magnitude and then used to rescale back to the original magnitude.

static_features (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) : Static features of each time series' in a batch which are copied to the covariates at inference time.

## SampleTSPredictionOutput[[transformers.modeling_outputs.SampleTSPredictionOutput]]

#### transformers.modeling_outputs.SampleTSPredictionOutput[[transformers.modeling_outputs.SampleTSPredictionOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1670)

Base class for time series model's predictions outputs that contains the sampled values from the chosen
distribution.

**Parameters:**

sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length)` or `(batch_size, num_samples, prediction_length, input_size)`) : Sampled values from the chosen distribution.
