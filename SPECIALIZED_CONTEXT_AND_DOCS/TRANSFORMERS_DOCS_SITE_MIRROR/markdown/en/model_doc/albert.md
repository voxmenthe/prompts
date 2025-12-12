# ALBERT

[ALBERT](https://huggingface.co/papers/1909.11942) is designed to address memory limitations of scaling and training of [BERT](./bert). It adds two parameter reduction techniques. The first, factorized embedding parametrization, splits the larger vocabulary embedding matrix into two smaller matrices so you can grow the hidden size without adding a lot more parameters. The second, cross-layer parameter sharing, allows layer to share parameters which keeps the number of learnable parameters lower.

ALBERT was created to address problems like -- GPU/TPU memory limitations, longer training times, and unexpected model degradation in BERT. ALBERT uses two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

- **Factorized embedding parameterization:** The large vocabulary embedding matrix is decomposed into two smaller matrices, reducing memory consumption.
- **Cross-layer parameter sharing:** Instead of learning separate parameters for each transformer layer, ALBERT shares parameters across layers, further reducing the number of learnable weights.

ALBERT uses absolute position embeddings (like BERT) so padding is applied at right. Size of embeddings is 128 While BERT uses 768. ALBERT can processes maximum 512 token at a time.

You can find all the original ALBERT checkpoints under the [ALBERT community](https://huggingface.co/albert) organization.

> [!TIP]
> Click on the ALBERT models in the right sidebar for more examples of how to apply ALBERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and from the command line.

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="albert-base-v2",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.", top_k=5)
```

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForMaskedLM.from_pretrained(
    "albert/albert-base-v2",
    dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto"
)

prompt = "Plants create energy through a process known as [MASK]."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    predictions = outputs.logits[0, mask_token_index]

top_k = torch.topk(predictions, k=5).indices.tolist()
for token_id in top_k[0]:
    print(f"Prediction: {tokenizer.decode([token_id])}")
```

```bash
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers run --task fill-mask --model albert-base-v2 --device 0
```

## Notes

- Inputs should be padded on the right because BERT uses absolute position embeddings.
- The embedding size `E` is different from the hidden size `H` because the embeddings are context independent (one embedding vector represents one token) and the hidden states are context dependent (one hidden state represents a sequence of tokens). The embedding matrix is also larger because `V x E` where `V` is the vocabulary size. As a result, it's more logical if `H >> E`. If `E 

- `AlbertForSequenceClassification` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).

- Check the [Text classification task guide](../tasks/sequence_classification) on how to use the model.

- `AlbertForTokenClassification` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification).

- [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Token classification task guide](../tasks/token_classification) on how to use the model.

- `AlbertForMaskedLM` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Masked language modeling task guide](../tasks/masked_language_modeling) on how to use the model.

- `AlbertForQuestionAnswering` is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Question answering task guide](../tasks/question_answering) on how to use the model.

**Multiple choice**

- [AlbertForMultipleChoice](/docs/transformers/main/en/model_doc/albert#transformers.AlbertForMultipleChoice) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- Check the [Multiple choice task guide](../tasks/multiple_choice) on how to use the model.

## AlbertConfig[[transformers.AlbertConfig]]

#### transformers.AlbertConfig[[transformers.AlbertConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/configuration_albert.py#L21)

This is the configuration class to store the configuration of a `AlbertModel`. It is used
to instantiate an ALBERT model according to the specified arguments, defining the model architecture. Instantiating
a configuration with the defaults will yield a similar configuration to that of the ALBERT
[albert/albert-xxlarge-v2](https://huggingface.co/albert/albert-xxlarge-v2) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import AlbertConfig, AlbertModel

>>> # Initializing an ALBERT-xxlarge style configuration
>>> albert_xxlarge_configuration = AlbertConfig()

>>> # Initializing an ALBERT-base style configuration
>>> albert_base_configuration = AlbertConfig(
...     hidden_size=768,
...     num_attention_heads=12,
...     intermediate_size=3072,
... )

>>> # Initializing a model (with random weights) from the ALBERT-base style configuration
>>> model = AlbertModel(albert_xxlarge_configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30000) : Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `AlbertModel`.

embedding_size (`int`, *optional*, defaults to 128) : Dimensionality of vocabulary embeddings.

hidden_size (`int`, *optional*, defaults to 4096) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_hidden_groups (`int`, *optional*, defaults to 1) : Number of groups for the hidden layers, parameters in the same group are shared.

num_attention_heads (`int`, *optional*, defaults to 64) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 16384) : The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.

inner_group_num (`int`, *optional*, defaults to 1) : The number of inner repetition of attention and ffn.

hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu_new"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large (e.g., 512 or 1024 or 2048).

type_vocab_size (`int`, *optional*, defaults to 2) : The vocabulary size of the `token_type_ids` passed when calling `AlbertModel`.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

classifier_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for attached classifiers.

pad_token_id (`int`, *optional*, defaults to 0) : Padding token id.

bos_token_id (`int`, *optional*, defaults to 2) : Beginning of stream token id.

eos_token_id (`int`, *optional*, defaults to 3) : End of stream token id.

## AlbertTokenizer

[[autodoc]] AlbertTokenizer - build_inputs_with_special_tokens - get_special_tokens_mask - create_token_type_ids_from_sequences - save_vocabulary

## AlbertTokenizerFast[[transformers.AlbertTokenizer]]

#### transformers.AlbertTokenizer[[transformers.AlbertTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/tokenization_albert.py#L31)

Construct a "fast" ALBERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods

**Parameters:**

do_lower_case (`bool`, *optional*, defaults to `True`) : Whether or not to lowercase the input when tokenizing.

keep_accents (`bool`, *optional*, defaults to `False`) : Whether or not to keep accents when tokenizing.

bos_token (`str`, *optional*, defaults to `"[CLS]"`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `"[SEP]"`) : The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `"[SEP]"`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `"[CLS]"`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `"[MASK]"`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.

trim_offsets (`bool`, *optional*, defaults to `True`) : Whether the post processing step should trim offsets to avoid including whitespaces.

vocab (`str` or `list[tuple[str, float]]`, *optional*) : Custom vocabulary with `(token, score)` tuples. If not provided, vocabulary is loaded from `vocab_file`.

vocab_file (`str`, *optional*) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that contains the vocabulary necessary to instantiate a tokenizer.

## Albert specific outputs[[transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput]]

#### transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput[[transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py#L331)

Output type of `AlbertForPreTraining`.

**Parameters:**

loss (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) : Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.

prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

sop_logits (`torch.FloatTensor` of shape `(batch_size, 2)`) : Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).

hidden_states (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## AlbertModel

[[autodoc]] AlbertModel - forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining - forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM - forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification - forward

## AlbertForMultipleChoice[[transformers.AlbertForMultipleChoice]]

#### transformers.AlbertForMultipleChoice[[transformers.AlbertForMultipleChoice]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py#L872)

The Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.AlbertForMultipleChoice.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py#L883[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) and
  [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
  num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
  *input_ids* above)0[transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput](/docs/transformers/main/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or `tuple(torch.FloatTensor)`A [transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput](/docs/transformers/main/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlbertConfig](/docs/transformers/main/en/model_doc/albert#transformers.AlbertConfig)) and inputs.

- **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
- **prediction_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **sop_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) -- Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
  before SoftMax).
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [AlbertForMultipleChoice](/docs/transformers/main/en/model_doc/albert#transformers.AlbertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, AlbertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-xxlarge-v2")
>>> model = AlbertForMultipleChoice.from_pretrained("albert/albert-xxlarge-v2")

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

config ([AlbertConfig](/docs/transformers/main/en/model_doc/albert#transformers.AlbertConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput](/docs/transformers/main/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput](/docs/transformers/main/en/model_doc/albert#transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlbertConfig](/docs/transformers/main/en/model_doc/albert#transformers.AlbertConfig)) and inputs.

- **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
- **prediction_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **sop_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) -- Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
  before SoftMax).
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification - forward

## AlbertForQuestionAnswering

[[autodoc]] AlbertForQuestionAnswering - forward
