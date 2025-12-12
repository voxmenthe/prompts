*This model was released on 2020-05-22 and added to Hugging Face Transformers on 2020-11-16.*

# RAG

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

[Retrieval-Augmented Generation (RAG)](https://huggingface.co/papers/2005.11401) combines a pretrained language model (parametric memory) with access to an external data source (non-parametric memory) by means of a pretrained neural retriever. RAG fetches relevant passages and conditions its generation on them during inference. This often makes the answers more factual and lets you update knowledge by changing the index instead of retraining the whole model.

You can find all the original RAG checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=rag) organization.

This model was contributed by [ola13](https://huggingface.co/ola13).

Click on the RAG models in the right sidebar for more examples of how to apply RAG to different language tasks.

The examples below demonstrates how to generate text with [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel).

AutoModel


```
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-token-nq",
    retriever=retriever,
    dtype="auto",
    attn_implementation="flash_attention_2",
)
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

Quantization reduces memory by storing weights in lower precision. See the [Quantization](../quantization/overview) overview for supported backends.
The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.


```
import torch
from transformers import BitsAndBytesConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-token-nq",
    retriever=retriever,
    quantization_config=bnb,   # quantizes generator weights
    device_map="auto",
)
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

## RagConfig

### class transformers.RagConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/configuration_rag.py#L80)

( vocab\_size = None is\_encoder\_decoder = True prefix = None bos\_token\_id = None pad\_token\_id = None eos\_token\_id = None decoder\_start\_token\_id = None title\_sep = ' / ' doc\_sep = ' // ' n\_docs = 5 max\_combined\_length = 300 retrieval\_vector\_size = 768 retrieval\_batch\_size = 8 dataset = 'wiki\_dpr' dataset\_split = 'train' index\_name = 'compressed' index\_path = None passages\_path = None use\_dummy\_dataset = False reduce\_loss = False label\_smoothing = 0.0 do\_deduplication = True exclude\_bos\_score = False do\_marginalize = False output\_retrieved = False use\_cache = True forced\_eos\_token\_id = None dataset\_revision = None \*\*kwargs  )

Parameters

* **title\_sep** (`str`, *optional*, defaults to `" / "`) —
  Separator inserted between the title and the text of the retrieved document when calling [RagRetriever](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever).
* **doc\_sep** (`str`, *optional*, defaults to `" // "`) —
  Separator inserted between the text of the retrieved document and the original input when calling
  [RagRetriever](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever).
* **n\_docs** (`int`, *optional*, defaults to 5) —
  Number of documents to retrieve.
* **max\_combined\_length** (`int`, *optional*, defaults to 300) —
  Max length of contextualized input returned by `__call__()`.
* **retrieval\_vector\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the document embeddings indexed by [RagRetriever](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever).
* **retrieval\_batch\_size** (`int`, *optional*, defaults to 8) —
  Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
  [RagRetriever](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever).
* **dataset** (`str`, *optional*, defaults to `"wiki_dpr"`) —
  A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
  using `datasets.list_datasets()`).
* **dataset\_split** (`str`, *optional*, defaults to `"train"`) —
  Which split of the `dataset` to load.
* **index\_name** (`str`, *optional*, defaults to `"compressed"`) —
  The index name of the index associated with the `dataset`. One can choose between `"legacy"`, `"exact"` and
  `"compressed"`.
* **index\_path** (`str`, *optional*) —
  The path to the serialized faiss index on disk.
* **passages\_path** (`str`, *optional*) —
  A path to text passages compatible with the faiss index. Required if using
  `LegacyIndex`
* **use\_dummy\_dataset** (`bool`, *optional*, defaults to `False`) —
  Whether to load a “dummy” variant of the dataset specified by `dataset`.
* **label\_smoothing** (`float`, *optional*, defaults to 0.0) —
  Only relevant if `return_loss` is set to `True`. Controls the `epsilon` parameter value for label smoothing
  in the loss calculation. If set to 0, no label smoothing is performed.
* **do\_marginalize** (`bool`, *optional*, defaults to `False`) —
  If `True`, the logits are marginalized over all documents by making use of
  `torch.nn.functional.log_softmax`.
* **reduce\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether or not to reduce the NLL loss using the `torch.Tensor.sum` operation.
* **do\_deduplication** (`bool`, *optional*, defaults to `True`) —
  Whether or not to deduplicate the generations from different context documents for a given input. Has to be
  set to `False` if used while training with distributed backend.
* **exclude\_bos\_score** (`bool`, *optional*, defaults to `False`) —
  Whether or not to disregard the BOS token when computing the loss.
* **output\_retrieved(`bool`,** *optional*, defaults to `False`) —
  If set to `True`, `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
  `context_attention_mask` are returned. See returned tensors for more detail.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **forced\_eos\_token\_id** (`int`, *optional*) —
  The id of the token to force as the last generated token when `max_length` is reached. Usually set to
  `eos_token_id`.

[RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig) stores the configuration of a *RagModel*. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and
can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_question\_encoder\_generator\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/configuration_rag.py#L172)

( question\_encoder\_config: PretrainedConfig generator\_config: PretrainedConfig \*\*kwargs  ) → [EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)

Returns

[EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)

An instance of a configuration object

Instantiate a [EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) (or a derived class) from a pre-trained encoder model configuration and
decoder model configuration.

## RagTokenizer

### class transformers.RagTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/tokenization_rag.py#L29)

( question\_encoder generator  )

## Rag specific outputs

### class transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L43)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None doc\_scores: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None retrieved\_doc\_embeds: typing.Optional[torch.FloatTensor] = None retrieved\_doc\_ids: typing.Optional[torch.LongTensor] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None question\_encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None question\_enc\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None question\_enc\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_enc\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None generator\_enc\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_enc\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_dec\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_dec\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_cross\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) —
  Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
  each vocabulary token.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

  Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
  (see `past_key_values` input) to speed up sequential decoding.
* **retrieved\_doc\_embeds** (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output\_retrieved=True*) —
  Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
  the `doc_scores`.
* **retrieved\_doc\_ids** (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output\_retrieved=True*) —
  The indexes of the embedded documents retrieved by the retriever.
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input ids post-processed from the retrieved documents and the question encoder input\_ids by the retriever.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.
* **question\_encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
  model.
* **question\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
* **question\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_enc\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
* **generator\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
* **generator\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_dec\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
* **generator\_dec\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for retriever augmented marginalized models outputs.

### class transformers.models.rag.modeling\_rag.RetrievAugLMOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L136)

( logits: typing.Optional[torch.FloatTensor] = None doc\_scores: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None retrieved\_doc\_embeds: typing.Optional[torch.FloatTensor] = None retrieved\_doc\_ids: typing.Optional[torch.LongTensor] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None question\_encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None question\_enc\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None question\_enc\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_enc\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None generator\_enc\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_enc\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_dec\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_dec\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None generator\_cross\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) —
  Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
  each vocabulary token.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

  Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
  (see `past_key_values` input) to speed up sequential decoding.
* **retrieved\_doc\_embeds** (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output\_retrieved=True*) —
  Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
  the `doc_scores`.
* **retrieved\_doc\_ids** (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output\_retrieved=True*) —
  The indexes of the embedded documents retrieved by the retriever.
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input ids post-processed from the retrieved documents and the question encoder input\_ids by the retriever.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.
* **question\_encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
  model.
* **question\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
* **question\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_enc\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
* **generator\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
* **generator\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_dec\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
* **generator\_dec\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

## RagRetriever

### class transformers.RagRetriever

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L349)

( config question\_encoder\_tokenizer generator\_tokenizer index = None init\_retrieval = True  )

Parameters

* **config** ([RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig)) —
  The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
  `Index` to build. You can load your own custom dataset with `config.index_name="custom"` or use a canonical
  one (default) from the datasets library with `config.index_name="wiki_dpr"` for example.
* **question\_encoder\_tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) —
  The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
  generator\_tokenizer.
* **generator\_tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) —
  The tokenizer used for the generator part of the RagModel.
* **index** (`Index`, optional, defaults to the one defined by the configuration) —
  If specified, use this index instead of the one built using the configuration

Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
contents, and it formats them to be used with a RagModel.

Examples:


```
>>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
>>> from transformers import RagRetriever

>>> retriever = RagRetriever.from_pretrained(
...     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
... )

>>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
>>> from transformers import RagRetriever

>>> dataset = (
...     ...
... )  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a supported index (e.g., Faiss or other index types depending on your setup)
>>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

>>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
>>> from transformers import RagRetriever

>>> dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
>>> index_path = "path/to/my/index"  # index saved via *dataset.get_index("embeddings").save(...)*
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/dpr-ctx_encoder-single-nq-base",
...     index_name="custom",
...     passages_path=dataset_path,
...     index_path=index_path,
... )

>>> # To load the legacy index built originally for Rag's paper
>>> from transformers import RagRetriever

>>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")
```

#### init\_retrieval

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L484)

( )

Retriever initialization function. It loads the index into memory.

#### postprocess\_docs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L492)

( docs input\_strings prefix n\_docs return\_tensors = None  ) → `tuple(tensors)`

Parameters

* **docs** (`dict`) —
  Retrieved documents.
* **input\_strings** (`str`) —
  Input strings decoded by `preprocess_query`.
* **prefix** (`str`) —
  Prefix added at the beginning of each input, typically used with T5-based models.

Returns

`tuple(tensors)`

a tuple consisting of two elements: contextualized `input_ids` and a compatible
`attention_mask`.

Postprocessing retrieved `docs` and combining them with `input_strings`.

#### retrieve

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L564)

( question\_hidden\_states: ndarray n\_docs: int  ) → `tuple[np.ndarray, np.ndarray, list[dict]]`

Parameters

* **question\_hidden\_states** (`np.ndarray` of shape `(batch_size, vector_size)`) —
  A batch of query vectors to retrieve with.
* **n\_docs** (`int`) —
  The number of docs retrieved per query.

Returns

`tuple[np.ndarray, np.ndarray, list[dict]]`

A tuple with the following objects:

* **retrieved\_doc\_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) — The retrieval embeddings
  of the retrieved docs per query.
* **doc\_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) — The ids of the documents in the index
* **doc\_dicts** (`list[dict]`): The `retrieved_doc_embeds` examples per query.

Retrieves documents for specified `question_hidden_states`.

## RagModel

### class transformers.RagModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L383)

( config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None question\_encoder: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None generator: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None retriever: typing.Optional[transformers.models.rag.retrieval\_rag.RagRetriever] = None \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **question\_encoder** (`PreTrainedModel`, *optional*) —
  The model responsible for encoding the question into hidden states for retrieval.
* **generator** (`PreTrainedModel`, *optional*) —
  The model responsible for generating text based on retrieved documents.
* **retriever** (`RagRetriever`, *optional*) —
  The component responsible for retrieving documents from a knowledge base given the encoded question.

The bare Rag Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L434)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None doc\_scores: typing.Optional[torch.FloatTensor] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_retrieved: typing.Optional[bool] = None n\_docs: typing.Optional[int] = None  ) → [transformers.models.rag.modeling\_rag.RetrievAugLMOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig), used to initialize the model, specifies
  which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
  obtain the indices.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
  *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs * sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
  generator’s encoder.

  Used by the ([RagModel](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel)) model during decoding.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Provide for generation tasks. `None` by default, construct as per instructions for the generator model
  you’re using with your RAG instance.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
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
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
  has to be provided to the forward pass. `doc_scores` can be computed via
  `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to
  the forward pass. `context_input_ids` are returned by `__call__()`.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be
  provided to the forward pass. `context_attention_mask` are returned by `__call__()`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_retrieved** (`bool`, *optional*) —
  Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
  `context_attention_mask`. See returned tensors for more detail.
* **n\_docs** (`int`, *optional*) —
  The number of documents to retrieve.

Returns

[transformers.models.rag.modeling\_rag.RetrievAugLMOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.rag.modeling\_rag.RetrievAugLMOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig)) and inputs.

* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
  each vocabulary token.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) — Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

  Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
  (see `past_key_values` input) to speed up sequential decoding.
* **retrieved\_doc\_embeds** (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output\_retrieved=True*) — Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
  the `doc_scores`.
* **retrieved\_doc\_ids** (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output\_retrieved=True*) — The indexes of the embedded documents retrieved by the retriever.
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) — Input ids post-processed from the retrieved documents and the question encoder input\_ids by the retriever.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) — Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.
* **question\_encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
  model.
* **question\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
* **question\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_enc\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
* **generator\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
* **generator\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_dec\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
* **generator\_dec\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [RagModel](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, RagRetriever, RagModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
... )
>>> # initialize with RagRetriever to do everything in one forward call
>>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

>>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
>>> outputs = model(input_ids=inputs["input_ids"])
```

## RagSequenceForGeneration

### class transformers.RagSequenceForGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L671)

( config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None question\_encoder: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None generator: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None retriever: typing.Optional[transformers.models.rag.retrieval\_rag.RagRetriever] = None \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **question\_encoder** (`PreTrainedModel`, *optional*) —
  The model responsible for encoding the question into hidden states for retrieval.
* **generator** (`PreTrainedModel`, *optional*) —
  The model responsible for generating text based on retrieved documents.
* **retriever** (`RagRetriever`, *optional*) —
  The component responsible for retrieving documents from a knowledge base given the encoded question.

A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L708)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.Tensor]]] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None doc\_scores: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_retrieved: typing.Optional[bool] = None exclude\_bos\_score: typing.Optional[bool] = None reduce\_loss: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None n\_docs: typing.Optional[int] = None \*\*kwargs  ) → [transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig), used to initialize the model, specifies
  which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
  obtain the indices.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
  *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs * sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
  generator’s encoder.

  Used by the ([RagModel](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel)) model during decoding.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Provide for generation tasks. `None` by default, construct as per instructions for the generator model
  you’re using with your RAG instance.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
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
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to
  the forward pass. `context_input_ids` are returned by `__call__()`.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be
  provided to the forward pass. `context_attention_mask` are returned by `__call__()`.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
  has to be provided to the forward pass. `doc_scores` can be computed via
  `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_retrieved** (`bool`, *optional*) —
  Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
  `context_attention_mask`. See returned tensors for more detail.
* **exclude\_bos\_score** (`bool`, *optional*) —
  Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing
  the loss.
* **reduce\_loss** (`bool`, *optional*) —
  Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
  operation.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **n\_docs** (`int`, *optional*) —
  The number of documents to retrieve.

Returns

[transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
  each vocabulary token.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) — Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

  Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
  (see `past_key_values` input) to speed up sequential decoding.
* **retrieved\_doc\_embeds** (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output\_retrieved=True*) — Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
  the `doc_scores`.
* **retrieved\_doc\_ids** (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output\_retrieved=True*) — The indexes of the embedded documents retrieved by the retriever.
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) — Input ids post-processed from the retrieved documents and the question encoder input\_ids by the retriever.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) — Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.
* **question\_encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
  model.
* **question\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
* **question\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_enc\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
* **generator\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
* **generator\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_dec\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
* **generator\_dec\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [RagSequenceForGeneration](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagSequenceForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
... )
>>> # initialize with RagRetriever to do everything in one forward call
>>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

>>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
>>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
>>> input_ids = inputs["input_ids"]
>>> labels = targets["input_ids"]
>>> outputs = model(input_ids=input_ids, labels=labels)

>>> # or use retriever separately
>>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
>>> # 1. Encode
>>> question_hidden_states = model.question_encoder(input_ids)[0]
>>> # 2. Retrieve
>>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
>>> doc_scores = torch.bmm(
...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
... ).squeeze(1)
>>> # 3. Forward to generator
>>> outputs = model(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
...     decoder_input_ids=labels,
... )
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L881)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None doc\_scores: typing.Optional[torch.FloatTensor] = None do\_deduplication: typing.Optional[bool] = None num\_return\_sequences: typing.Optional[int] = None num\_beams: typing.Optional[int] = None n\_docs: typing.Optional[int] = None \*\*model\_kwargs  ) → `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  The sequence used as a prompt for the generation. If `input_ids` is not passed, then
  `context_input_ids` has to be provided.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input IDs post-processed from the retrieved documents and the question encoder input\_ids by the
  retriever.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.

  If the model is not initialized with a `retriever` or `input_ids` is not given, `context_input_ids` and
  `context_attention_mask` have to be provided to the forward pass. They are returned by
  `__call__()`.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.

  If the model is not initialized with a `retriever` or `input_ids` is not given, `doc_scores` has to be
  provided to the forward pass. `doc_scores` are returned by `__call__()`.
* **do\_deduplication** (`bool`, *optional*) —
  Whether or not to deduplicate the generations from different context documents for a given input. Has
  to be set to `False` if used while training with distributed backend.
* **num\_return\_sequences(`int`,** *optional*, defaults to 1) —
  The number of independently computed returned sequences for each element in the batch. Note that this
  is not the value we pass to the `generator`’s `[generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate)` function,
  where we set `num_return_sequences` to `num_beams`.
* **num\_beams** (`int`, *optional*, defaults to 1) —
  Number of beams for beam search. 1 means no beam search.
* **n\_docs** (`int`, *optional*, defaults to `config.n_docs`) —
  Number of documents to retrieve and/or number of documents for which to generate an answer.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional kwargs will be passed to [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate).

Returns

`torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

The generated
sequences. The second dimension (sequence length) is either equal to `max_length` or shorter if all batches
finished early due to the `eos_token_id`.

Implements RAG sequence “thorough” decoding. Read the [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate)` documentation
for more information on how to set other generate input parameters.

## RagTokenForGeneration

### class transformers.RagTokenForGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L1108)

( config: typing.Optional[transformers.configuration\_utils.PretrainedConfig] = None question\_encoder: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None generator: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None retriever: typing.Optional[transformers.models.rag.retrieval\_rag.RagRetriever] = None \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **question\_encoder** (`PreTrainedModel`, *optional*) —
  The model responsible for encoding the question into hidden states for retrieval.
* **generator** (`PreTrainedModel`, *optional*) —
  The model responsible for generating text based on retrieved documents.
* **retriever** (`RagRetriever`, *optional*) —
  The component responsible for retrieving documents from a knowledge base given the encoded question.

A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L1220)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.Tensor]]] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None doc\_scores: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_retrieved: typing.Optional[bool] = None do\_marginalize: typing.Optional[bool] = None reduce\_loss: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None n\_docs: typing.Optional[int] = None \*\*kwargs  ) → [transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. [RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig), used to initialize the model, specifies
  which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
  obtain the indices.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
  *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs * sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
  generator’s encoder.

  Used by the ([RagModel](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel)) model during decoding.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Provide for generation tasks. `None` by default, construct as per instructions for the generator model
  you’re using with your RAG instance.
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
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
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to
  the forward pass. `context_input_ids` are returned by `__call__()`.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be
  provided to the forward pass. `context_attention_mask` are returned by `__call__()`.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
  has to be provided to the forward pass. `doc_scores` can be computed via
  `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_retrieved** (`bool`, *optional*) —
  Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
  `context_attention_mask`. See returned tensors for more detail.
* **do\_marginalize** (`bool`, *optional*) —
  If `True`, the logits are marginalized over all documents by making use of
  `torch.nn.functional.log_softmax`.
* **reduce\_loss** (`bool`, *optional*) —
  Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
  operation.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **n\_docs** (`int`, *optional*) —
  The number of documents to retrieve.

Returns

[transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.rag.modeling\_rag.RetrievAugLMMarginOutput](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
  each vocabulary token.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) — Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

  Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
  (see `past_key_values` input) to speed up sequential decoding.
* **retrieved\_doc\_embeds** (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output\_retrieved=True*) — Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
  the `doc_scores`.
* **retrieved\_doc\_ids** (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output\_retrieved=True*) — The indexes of the embedded documents retrieved by the retriever.
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) — Input ids post-processed from the retrieved documents and the question encoder input\_ids by the retriever.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) — Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.
* **question\_encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
  model.
* **question\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
* **question\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_enc\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
* **generator\_enc\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
* **generator\_enc\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_dec\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
* **generator\_dec\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **generator\_cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [RagTokenForGeneration](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagTokenForGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
>>> retriever = RagRetriever.from_pretrained(
...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
... )
>>> # initialize with RagRetriever to do everything in one forward call
>>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

>>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
>>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
>>> input_ids = inputs["input_ids"]
>>> labels = targets["input_ids"]
>>> outputs = model(input_ids=input_ids, labels=labels)

>>> # or use retriever separately
>>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
>>> # 1. Encode
>>> question_hidden_states = model.question_encoder(input_ids)[0]
>>> # 2. Retrieve
>>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
>>> doc_scores = torch.bmm(
...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
... ).squeeze(1)
>>> # 3. Forward to generator
>>> outputs = model(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
...     decoder_input_ids=labels,
... )

>>> # or directly generate
>>> generated = model.generate(
...     context_input_ids=docs_dict["context_input_ids"],
...     context_attention_mask=docs_dict["context_attention_mask"],
...     doc_scores=doc_scores,
... )
>>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L1393)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None context\_input\_ids: typing.Optional[torch.LongTensor] = None context\_attention\_mask: typing.Optional[torch.LongTensor] = None doc\_scores: typing.Optional[torch.FloatTensor] = None n\_docs: typing.Optional[int] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None prefix\_allowed\_tokens\_fn: typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None logits\_processor: typing.Optional[transformers.generation.logits\_process.LogitsProcessorList] = [] stopping\_criteria: typing.Optional[transformers.generation.stopping\_criteria.StoppingCriteriaList] = [] \*\*kwargs  ) → `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  The sequence used as a prompt for the generation. If `input_ids` is not passed, then
  `context_input_ids` has to be provided.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **context\_input\_ids** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.

  If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
  forward pass. `context_input_ids` are returned by `__call__()`.
* **context\_attention\_mask** (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output\_retrieved=True*) —
  Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
  retriever.

  If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
  forward pass. `context_input_ids` are returned by `__call__()`.
* **doc\_scores** (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`) —
  Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
  `question_encoder_last_hidden_state`.

  If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
  forward pass. `context_input_ids` are returned by `__call__()`.
* **n\_docs** (`int`, *optional*, defaults to `config.n_docs`) —
  Number of documents to retrieve and/or number of documents for which to generate an answer.
* **generation\_config** (`~generation.GenerationConfig`, *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which has the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) —
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
  `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
  the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
  constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and a
  model’s config. If a logit processor is passed that is already created with the arguments or a model’s
  config an error is thrown.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complement the default stopping criteria built from arguments and a
  model’s config. If a stopping criteria is passed that is already created with the arguments or a
  model’s config an error is thrown.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model.

Returns

`torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

The generated
sequences. The second dimension (sequence\_length) is either equal to `max_length` or shorter if all batches
finished early due to the `eos_token_id`.

Implements RAG token decoding.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/rag.md)
